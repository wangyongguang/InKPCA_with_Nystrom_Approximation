"""
    subspace classes
    CovarianceSpace: covariance subspace
    PCASpace: PCA subspace 
    FreqDirSpace: Frequent Directions Space
"""

import abc

import torch
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn.decomposition.pca import _assess_dimension_
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import sys
import data_IncrKPCA
from incremental_kpca_milton_Real import IncrKPCA_real,nystrom_approximation
from kernels import kernel_matrix,rbf,adjust_K,median_distance
Kernels= ['additive_chi2',
    'chi2',
    'linear',
    'polynomial',
    'poly',
    'rbf',
    'laplacian',
    'sigmoid',
    'cosine']

class Subspace(torch.nn.Module, metaclass=abc.ABCMeta):
    subclasses = {}

    @classmethod
    def register_subclass(cls, subspace_type):
        def decorator(subclass):
            cls.subclasses[subspace_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, subspace_type, **kwargs):
        if subspace_type not in cls.subclasses:
            raise ValueError('Bad subspaces type {}'.format(subspace_type))
        return cls.subclasses[subspace_type](**kwargs)

    def __init__(self):
        super(Subspace, self).__init__()

    @abc.abstractmethod
    def collect_vector(self, vector):
        pass

    @abc.abstractmethod
    def get_space(self):
        pass


@Subspace.register_subclass('random')
class RandomSpace(Subspace):
    def __init__(self, num_parameters, rank=20, method='dense'):
        assert method in ['dense', 'fastfood']

        super(RandomSpace, self).__init__()

        self.num_parameters = num_parameters
        self.rank = rank
        self.method = method

        if method == 'dense':
            self.subspace = torch.randn(rank, num_parameters)

        if method == 'fastfood':
            raise NotImplementedError("FastFood transform hasn't been implemented yet")

    # random subspace is independent of data
    def collect_vector(self, vector):
        pass
    
    def get_space(self):
        return self.subspace



@Subspace.register_subclass('covariance')
class CovarianceSpace(Subspace):

    def __init__(self, num_parameters, max_rank=20):
        super(CovarianceSpace, self).__init__()

        self.num_parameters = num_parameters

        self.register_buffer('rank', torch.zeros(1, dtype=torch.long))
        self.register_buffer('cov_mat_sqrt',
                             torch.empty(0, self.num_parameters, dtype=torch.float32))
        self.register_buffer("X_tilde",
                             torch.empty((0,0),dtype=torch.float32))

        self.max_rank = max_rank
        # print("self.cov_mat_sqrt.shape=", self.cov_mat_sqrt.shape)

    def collect_vector(self, vector):
        #刚开始收集vector=w-self.mean时就比较rank+1与max_rank的大小；
        # 如果超过max_rank就将第二行往后的vector提取出来，目的是给新的一个vector留位置
        if self.rank.item() + 1 > self.max_rank:
            #self.rank为具体的某个空间类的自有变量，如类PCASpace存在自变量pca_rank=20
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :].cpu()
        #下面这句话的目的是将vector = w-self.mean 按行 添加 到开方的协方差矩阵中即 self.cov_mat_sqrt
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt.cpu(), vector.view(1, -1)), dim=0)
        #每添加一行，self.rank就加1，并与max_rank比较大小，self.rank不能超过max_rank
        self.rank = torch.min(self.rank.cpu() + 1, torch.as_tensor(self.max_rank).cpu()).view(-1)
        # print("self.cov_mat_sqrt.shape=", self.cov_mat_sqrt.shape)

        X = self.cov_mat_sqrt.clone().cpu().numpy() #在collect_vector()=X的同时对X进行IncrKPCA处理得到X_tilde
        # n = X.shape[0]
        inc = IncrKPCA_real(X, m0=self.m0,mmax=self.mmax,kernel=self.kernel, nystrom=True)
        # K = kernel_matrix(X, self.kernel, range(n),range(n))
        for i, L, U, L_nys, U_nys in inc:
            X_tilde = torch.FloatTensor(np.dot(U_nys, np.diag(np.sqrt(L_nys))))
            self.X_tilde = X_tilde


    def get_space(self):
        return self.cov_mat_sqrt.clone() / (self.cov_mat_sqrt.size(0) - 1) ** 0.5

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        rank = state_dict[prefix + 'rank'].item()

        self.cov_mat_sqrt = self.cov_mat_sqrt.new_empty((rank-1, self.cov_mat_sqrt.size()[1]))

        # self.X_tilde = self.X_tilde.new_empty((rank,rank-1))
        self.X_tilde = self.X_tilde.new_empty((rank-1, self.cov_mat_sqrt.size()[1]))

        super(CovarianceSpace, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                                           strict, missing_keys, unexpected_keys,
                                                           error_msgs)



# X_tilde1= torch.empty((0, 0), dtype=torch.float32)
@Subspace.register_subclass('InKPCA')
class IncrKPCASpace_real(CovarianceSpace):
    # global X_tilde1

    def __init__(self,num_parameters,m0=100,mmax=200,kernel='rbf',datasize=None,dataset=None):
        super(IncrKPCASpace_real,self).__init__(num_parameters)

        self.num_parameters = num_parameters

        self.register_buffer('cov_mat_sqrt',torch.zeros((0,self.num_parameters),dtype=torch.float32))
        self.register_buffer('rank', torch.zeros(1, dtype=torch.long))
        self.register_buffer("X_tilde",torch.empty((0,0),dtype=torch.float32))

        self.m0 = m0 #表示vector的最小收集数
        self.kernel = kernel
        self.mmax = mmax
        self.datasize = datasize
        self.dataset = dataset
        # self.kernel=kernel

    def collect_vector(self,vector):
        """
                如果没有下面两行会报如下错误：
                RuntimeError: Error(s) in loading state_dict for SWAG:
        	size mismatch for subspace.cov_mat_sqrt: copying a param with shape torch.Size([9, 15801])
        	from checkpoint, the shape in current model is torch.Size([0, 15801]).
                """
        #下面这句话的目的是将vector = w-self.mean 按行添加 到开方的协方差矩阵中即 self.cov_mat_sqrt

        if self.rank.item()+1 >= self.mmax: #这句话不应该在IncrKPCA中存在#self.rank为具体的某个空间类的自有变量，如类PCASpace存在自变量pca_rank=20
            # self.cov_mat_sqrt = self.get_space()
            # 因为上一句get_space()=X_tilde=U_nys*L_nys排好序了，最后一行协方差代表L_nys最小
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :] #所以可以去掉最后一行，加入新的vector到最后

            # self.cov_mat_sqrt = self.cov_mat_sqrt[1:,:]
            # self.X_tilde = self.X_tilde[:,:].cpu()
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt.cpu(), vector.view(1, -1)), dim=0)
        # print("milton-self.cov_mat_sqrt.shape=", self.cov_mat_sqrt.shape)
        self.rank = torch.min(self.rank.cpu() + 1, torch.as_tensor(self.mmax).cpu()).view(-1)
        # self.rank = self.rank.cpu()+1
        # self.X_tilde = self.X_tilde
        # print("milton-self.rank=", self.rank.item())
        # U_tilde = None
        # L_tilde = None
        X_tilde = torch.empty(0, 0, dtype=torch.float32)
        # global X_tilde1
        # global X_tilde_real
        # X_tilde_real = torch.empty(0, 0, dtype=torch.float32)
        fnorms = None
        X = self.cov_mat_sqrt.clone().cpu().numpy() #在collect_vector()=X的同时对X进行IncrKPCA处理得到X_tilde
        # n = X.shape[0]
        m, n = X.shape[0], X.shape[1]
        if self.kernel in Kernels and X.shape[0] > self.m0:
            # print("X=self.cov_mat_sqrt = ",self.cov_mat_sqrt)
            inc = IncrKPCA_real(X, m0=self.m0,mmax=self.mmax,kernel=self.kernel, nystrom=True,adjust=False)
            # K = kernel_matrix(X, self.kernel, range(n),range(n))
            K = kernel_matrix(X, self.kernel, range(m), range(m))
            for i, L, U, L_nys, U_nys in inc:
                K_tilde = np.dot(U_nys, np.dot(np.diag(L_nys), U_nys.T))
                fnorms = np.sqrt(np.sum(np.sum(np.power(K - K_tilde, 2))))
                # print("K = ",K)
                # print("K[0][1]=",K[0][1])
                # print("K_tilde = ",K_tilde)
                # print("K_tilde[0][1]=",K_tilde[0][1])
                # fnorms.append(fnorm) #SWAG在每一个epoch的collect_model过程中有subspace.collect_vector()的
                # fnorms = np.unique(fnorms) #去除列表重复的数据
                # rank = np.array([len(L_nys)])
                # self.rank = torch.from_numpy(rank)
                #有个问题是：每一个L_nys,U_nys生成一个X_tilde矩阵，
                # X_tilde1 = torch.FloatTensor(np.dot(U_nys, np.diag(np.sqrt(L_nys)))) # 这里要将L_nys取根号
            #     U_tilde = U_nys
            #     L_tilde = L_nys
            # #提出for循环是为了提高速度
            # X_tilde = np.dot(U_tilde, np.diag(np.sqrt(L_tilde)))
            X_tilde = np.dot(U_nys, np.diag(np.sqrt(L_nys)))
            k = n - X_tilde.shape[1]
            # c = np.zeros(m)
            c = [[0] * k] * m  # 用0替代NaN
            c = np.reshape(c, (m, k))  # 上面m,k怎么*都行，只要在这里reshape成(m,k)即可
            # for i in range(k): #不用for循环是因为当k很大时，太慢
            # print("X.shape=",X.shape)
            # print("self.X_tilde.shape=",self.X_tilde.shape)
            X_tilde = np.c_[X_tilde, c]
            X_tilde = np.nan_to_num(X_tilde)
            self.X_tilde = torch.FloatTensor(X_tilde)
                # X_tilde = torch.FloatTensor(np.dot(U_nys, np.diag(np.sqrt(L_nys))))

                # m, n = X.shape[0], X.shape[1]
                # k = n - X_tilde.shape[1]
                # # c = np.zeros(m)
                # c = [[0] * k] * m  # 0用1e-100替代
                # c = np.reshape(c, (m, k))  # 上面m,k怎么*都行，只要在这里reshape成(m,k)即可
                # # for i in range(k): #不用for循环是因为当k很大时，太慢
                # # print("X.shape=",X.shape)
                # # print("self.X_tilde.shape=",self.X_tilde.shape)
                # X_tilde = np.c_[X_tilde, c]
                # X_tilde = np.nan_to_num(X_tilde)
                #
                # self.X_tilde = torch.FloatTensor(X_tilde)
                # X_tilde = np.dot(U_nys, np.diag(np.sqrt(L_nys)))
                # k = n - X_tilde.shape[1]
                # # c = np.zeros(m)
                # c = [[1e-100] * k] * m  # 0用1e-100替代
                # c = np.reshape(c, (m, k))  # 上面m,k怎么*都行，只要在这里reshape成(m,k)即可
                # # for i in range(k): #不用for循环是因为当k很大时，太慢
                # # print("X.shape=",X.shape)
                # # print("self.X_tilde.shape=",self.X_tilde.shape)
                # X_tilde = np.c_[X_tilde, c]
                # X_tilde = np.nan_to_num(X_tilde)
                #
                # # X_tilde = torch.FloatTensor(np.dot(U_nys, np.diag(np.sqrt(L_nys))))
                #

                # self.X_tilde = torch.   cat((self.X_tilde.cpu(), self.X_tilde.view(1, -1)), dim=0)
        # print("X_tilde.size=", self.X_tilde.size())
        return fnorms

    # X_tilde2 = X_tilde1
    def get_space(self):
        # global X_tilde_real
        # X_tilde_real = torch.empty(0, 0, dtype=torch.float32)

        X = self.cov_mat_sqrt.clone().cpu().numpy()

        # print("X.shape=self.cov_mat_sqrt.clone.cpu().numpy().shape=", X.shape)
        # global X_tilde1 #表示使用的是全局变量X_tilde1;
        # 而collect_vector()中定义的：global X_tilde1----》也表示的是使用全局变量X_tilde1
        # 在collect_vector()将全局变量X_tilde1进行了重定义：
        # X_tilde1=torch.FloatTensor(np.dot(U_nys, np.diag(np.sqrt(L_nys))))
        # 意思是：在collect_vector()对全局变量X_tilde1进行重新赋值；在get_space()中继续将X_tilde1定义为
        # 全局变量；那么本函数中的X_tilde1就为那个值
        # X_tilde = X_tilde1.clone().cpu().numpy()
        # print("X_tilde.shape=", X_tilde.shape)
        # print("self.X_tilde.size = ", self.X_tilde.shape)# self.X_tilde.size = torch.Size([49, 48])
        # print("self.rank=", self.rank)
        # print("self.cov_mat_sqrt.shape=", self.cov_mat_sqrt.shape)
        # print("self.X_tilde.clone().cpu().numpy().shape=",self.X_tilde.clone().cpu().numpy().shape)
        X_tilde = self.X_tilde.clone().cpu().numpy()

        # print("self.X_tilde_real.shape=",X_tilde.shape)

         #继续将X_tilde中的nan和inf转换成相应的数值，如用0替代nan;有限数字代替inf

        cov_mat_sqrt_np = X_tilde
        cov_mat_sqrt_np /= (max(1, self.rank.item()) - 1) ** 0.5  # self.rank是len(L_nys)
        # print("get_space() and self.rank.item()=",self.rank.item())
        # print("get_space() and self.cov")
        return torch.FloatTensor(cov_mat_sqrt_np)
        # return torch.FloatTensor(self.cov_mat_sqrt.cpu()/((self.rank.item())-1)**0.5)
        #X_tilde填充0的部分使dev_vector用X_tilde更加有效，dev_vector方差可控


@Subspace.register_subclass('kpca')
class KPCASpace(CovarianceSpace):
    def __init__(self,num_parameters,kpca_rank=20,max_rank=20,kernel='rbf'):
        super(KPCASpace,self).__init__(num_parameters)
        self.kpca_rank=kpca_rank
        self.kernel = kernel
        self.max_rank = max_rank
        self.num_parameters = num_parameters
        self.kernel = kernel
        # self.register_buffer('cov_mat_sqrt', torch.empty(0, self.num_parameters, dtype=torch.float32))
        # self.register_buffer('rank', torch.zeros(1, dtype=torch.long))

    def collect_vector(self, vector):
        #刚开始收集vector=w-self.mean时就比较rank+1与max_rank的大小；
        # 如果超过max_rank就将第二行往后的vector提取出来，目的是给新的一个vector留位置
        if self.rank.item() + 1 > self.max_rank:
            #self.rank为具体的某个空间类的自有变量，如类PCASpace存在自变量pca_rank=20
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :].cpu()
        #下面这句话的目的是将vector = w-self.mean 按行 添加 到开方的协方差矩阵中即 self.cov_mat_sqrt
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt.cpu(), vector.view(1, -1)), dim=0)
        #每添加一行，self.rank就加1，并与max_rank比较大小，self.rank不能超过max_rank
        self.rank = torch.min(self.rank.cpu() + 1, torch.as_tensor(self.max_rank).cpu()).view(-1)
        print("self.cov_mat_sqrt.shape=", self.cov_mat_sqrt.shape)

    def get_space(self):
        print("begin get_space(): self.cov_mat_sqrt.shape=",self.cov_mat_sqrt.shape)
        cov_mat_sqrt_np = self.cov_mat_sqrt.clone().t().cpu().numpy()#这里跟PCA不同的是进行了一次转置.t(),变成(n_features,n_samples)
        #perform KPCA on DD'
        cov_mat_sqrt_np /=(max(1,self.rank.item())-1)**0.5

        kpca_rank = self.kpca_rank
        kpca_rank = max(1,min(kpca_rank,self.rank.item()))

        if self.kernel in Kernels:
            kernel = self.kernel
            kpca = KernelPCA(n_components=kpca_rank,kernel=kernel,gamma=15)
            # fit_transformed(X)返回的是：X_transformed = self.alphas_ * np.sqrt(self.lambdas_)
            #即cov_mat_sqrt_np通过kpca先核化K，然后通过对K通过 linalg.eigh（K）得到 self.alphas_和self.lambdas_-->
            # Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
            #所以X_transformed = self.alphas_ * np.sqrt(self.lambdas_);----->实验后发现这一行与下一行求X_transformed的结果是一样的
            #但是在求fit_inverse_transform时：X_transformed = np.dot(self.alphas_, np.diag(np.sqrt(self.lambdas_)))
            # 但self.alphas_ *np.dot(self.lambdas_)报错不能广播，这种形式是错误的
            #self._fit_inverse_transform(X_transformed, X)；上面这两句话的意思是如果要还原X_transformed时需要将lambda_
            #对角矩阵化然后与对应的特征向量相乘
            print("Before KernelPCA: self.cov_mat_sqrt.shape=",self.cov_mat_sqrt.shape)
            cov_kpca = kpca.fit_transform(cov_mat_sqrt_np)
            cov_kpca_t = torch.FloatTensor(cov_kpca.transpose()) #返回的时候将其转置回来(n_samples,n_features)
            print("After KernlePCA: cov_kpca_t.shape=",cov_kpca_t.shape)
            return cov_kpca_t
        else:
            print("There is no %s in Kernels! please choose an another kernel!"%self.kernel)

@Subspace.register_subclass('pca')
class PCASpace(Subspace): #PCASpace继承自CovarianceSpace，重新实现其中的函数，但collect_vector保留

    def __init__(self, num_parameters,kernel='rbf', pca_rank=20, max_rank=20):
        super(PCASpace, self).__init__()

        self.num_parameters = num_parameters
        self.max_rank = max_rank

        self.register_buffer('cov_mat_sqrt', torch.zeros((0, self.num_parameters), dtype=torch.float32))
        self.register_buffer('rank', torch.zeros(1, dtype=torch.long))
        # better phrasing for this condition?

        assert(pca_rank == 'mle' or isinstance(pca_rank, int))
        if pca_rank != 'mle':
            assert 1 <= pca_rank <= max_rank

        self.kernel = kernel
        self.pca_rank = pca_rank

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        rank = state_dict[prefix + 'rank'].item()

        self.cov_mat_sqrt = self.cov_mat_sqrt.new_empty((rank, self.cov_mat_sqrt.size()[1]))

        # self.X_tilde = self.X_tilde.new_empty((rank,rank-1))

        super(Subspace, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                                           strict, missing_keys, unexpected_keys,
                                                           error_msgs)
    def collect_vector(self, vector):
        # 刚开始收集vector=w-self.mean时就比较rank+1与max_rank的大小；
        # 如果超过max_rank就将第二行往后的vector提取出来，目的是给新的一个vector留位置
        if self.rank.item() + 1 > self.max_rank:
            # self.rank为具体的某个空间类的自有变量，如类PCASpace存在自变量pca_rank=20
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :].cpu()
        # 下面这句话的目的是将vector = w-self.mean 按行 添加 到开方的协方差矩阵中即 self.cov_mat_sqrt
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt.cpu(), vector.view(1, -1)), dim=0)
        # 每添加一行，self.rank就加1，并与max_rank比较大小，self.rank不能超过max_rank
        self.rank = torch.min(self.rank.cpu() + 1, torch.as_tensor(self.max_rank).cpu()).view(-1)
        # print("self.cov_mat_sqrt.shape=", self.cov_mat_sqrt.shape)

        fnorm = None
        cov_mat_sqrt_np = self.cov_mat_sqrt.clone().cpu().numpy()
        # perform PCA on DD'
        cov_mat_sqrt_np /= (max(1, self.rank.item() - 1)) ** 0.5
        if self.pca_rank == 'mle':
            pca_rank = self.rank.item()
        else:
            pca_rank = self.pca_rank
        #
        pca_rank = max(1, min(pca_rank, self.rank.item())) #pca_rank随self.rank而增大
        #truncated_svd.py中有类TruncatedSVD：
        # __init__(self, n_components=2, algorithm="randomized", n_iter=5,random_state=None, tol=0.):
        pca_decomp = TruncatedSVD(n_components=pca_rank) #TruncatedSVD主要对PCA的补充，要处理的是稀疏矩阵
        #TruncatedSVD类包括函数fit->fit_transform->返回X_new
        # K_tilde_1 = np.dot(np.diag(s), U.T)
        # K_tilde_2 = np.dot(U,K_tilde_1)
        n = cov_mat_sqrt_np.shape[0]
        if self.kernel in Kernels and n>= self.max_rank:
            # print("cov_mat_sqrt_np.shape[0] = ",cov_mat_sqrt_np.shape[0])
            # print("pca_rank=",pca_rank)
            pca_decomp.fit(cov_mat_sqrt_np) #return X_transformed = U * Sigma
            K = kernel_matrix(cov_mat_sqrt_np, self.kernel, range(n), range(n))
            # ss = StandardScaler()
            # std_K = ss.fit_transform(K)
            #PCA对数据矩阵进行了均值化
            U, s, Vt = randomized_svd(cov_mat_sqrt_np, n_components=pca_rank, n_iter=5)
            # K_tilde = np.dot(U, np.dot(np.diag(s), U.T))
            # cov_mat_sqrt_np_transformed = U*s
            #K_tilde中的cov_mat_sqrt_np用(s*Vt).T来替代，是因为在swag.sample()阶段调用get_space()返回的
            #是torch.FloatTensor(s[:, None] * Vt)
            K_tilde = kernel_matrix((np.dot(np.diag(s),Vt)).transpose(),self.kernel,range(n),range(n))
            # print("cov_mat_sqrt_np=",cov_mat_sqrt_np)
            # print("cov_mat_sqrt_np[0][1]=",cov_mat_sqrt_np[0][1])
            #
            # print("K = ",K)
            # print("K[0][1]=", K[0][1])
            # print("K_tilde.shape = ", K_tilde.shape)
            # print("K_tilde[0][1]=", K_tilde[0][1])
        # m, n = cov_mat_sqrt_np.shape[0], cov_mat_sqrt_np.shape[1]
        # k = n - K.shape[1]
        # c = np.zeros(m)
        # for i in range(k):
        #     K = np.c_[K, c]
        # print("cov_mat_sqrt_np.shape=", cov_mat_sqrt_np.shape)
        # print("K_tilde.shape=", K_tilde.shape)
        # print("K.shape=",K.shape)
            fnorm = np.sqrt(np.sum(np.sum(np.power(K - K_tilde, 2))))
        return fnorm

    def get_space(self):
        #self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt.cpu(), vector.view(1, -1)), dim=0)
        #实验中首先进行的是swag_model.collect_vector()获得vector=w-mean;因为pca_rank=20，所以
        #cov_mat_sqrt只收集到3000个epoch的最后20个epoch的vector
        cov_mat_sqrt_np = self.cov_mat_sqrt.clone().cpu().numpy()

        # perform PCA on DD'
        cov_mat_sqrt_np /= (max(1, self.rank.item() - 1))**0.5

        if self.pca_rank == 'mle':
            pca_rank = self.rank.item()
        else:
            pca_rank = self.pca_rank
        #
        pca_rank = max(1, min(pca_rank, self.rank.item()))
        #truncated_svd.py中有类TruncatedSVD：
        # __init__(self, n_components=2, algorithm="randomized", n_iter=5,random_state=None, tol=0.):
        pca_decomp = TruncatedSVD(n_components=pca_rank) #TruncatedSVD主要对PCA的补充，要处理的是稀疏矩阵
        #TruncatedSVD类包括函数fit->fit_transform->返回X_new
        pca_decomp.fit(cov_mat_sqrt_np)
        #如果在上面的pca_decomp中定义algorithm="randomized"则返回的是：
        #U, Sigma, VT = randomized_svd（）以及fit->fit_transform返回
        # X_transformed = U * Sigma= pca_decomp.fit(cov_mat_sqrt_np)
        # 我们需要s,Vt，所以单独运行randomized_svd, 上面pca_decomp.fit(cov_mat_sqrt_np)只是类实例化作用
        #当n_components=2时，s.shape=(2,); s[:,None].shape=(2,1)
        U, s, Vt = randomized_svd(cov_mat_sqrt_np, n_components=pca_rank, n_iter=5)

        # perform post-selection fitting
        if self.pca_rank == 'mle':#通过最大log似然算法来选择合适的rank
            eigs = s ** 2.0
            ll = np.zeros(len(eigs))
            correction = np.zeros(len(eigs))

            # compute minka's PCA marginal log likelihood and the correction term
            for rank in range(len(eigs)):
                # secondary correction term based on the rank of the matrix + degrees of freedom
                m = cov_mat_sqrt_np.shape[1] * rank - rank * (rank + 1) / 2.
                correction[rank] = 0.5 * m * np.log(cov_mat_sqrt_np.shape[0])
                ll[rank] = _assess_dimension_(spectrum=eigs,
                                              rank=rank,
                                              n_features=min(cov_mat_sqrt_np.shape),
                                              n_samples=max(cov_mat_sqrt_np.shape))
            
            self.ll = ll #log-likelihood
            self.corrected_ll = ll - correction
            self.pca_rank = np.nanargmax(self.corrected_ll)
            #Return the indices of the maximum values in the specified axis ignoring NaNs.
            print('PCA Rank is: ', self.pca_rank)
            return torch.FloatTensor(s[:self.pca_rank, None] * Vt[:self.pca_rank, :])
        else:
            return torch.FloatTensor(s[:, None] * Vt)
            #这句话的主要目的是将高维的cov_mat_sqrt_np=vector=mean-self.mean分解成低维的s[:,None]*Vt,并应用得到的这个低维矩阵
            #s作为特征值，每一个特征值对应一个Vt矩阵，
            #若s.shape=(pca_rank=2,),则s[:,None].shape=(2,1)
            #s=array([10,100]), Vt=array([[1,2,3],[1,2,3]])
            #则s[:,None]*Vt = array([[10],[100]])*Vt = array([[10,20,30],[100,200,300]])


@Subspace.register_subclass('freq_dir')
class FreqDirSpace(CovarianceSpace):
    def __init__(self, num_parameters, max_rank=20):
        super(FreqDirSpace, self).__init__(num_parameters, max_rank=max_rank)
        self.register_buffer('num_models', torch.zeros(1, dtype=torch.long))
        self.delta = 0.0
        self.normalized = False

    def collect_vector(self, vector):
        if self.rank >= 2 * self.max_rank:
            sketch = self.cov_mat_sqrt.numpy()
            [_, s, Vt] = np.linalg.svd(sketch, full_matrices=False)
            if s.size >= self.max_rank:
                current_delta = s[self.max_rank - 1] ** 2
                self.delta += current_delta
                s = np.sqrt(s[:self.max_rank - 1] ** 2 - current_delta)
            self.cov_mat_sqrt = torch.from_numpy(s[:, None] * Vt[:s.size, :])

        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
        self.rank = torch.as_tensor(self.cov_mat_sqrt.size(0))
        self.num_models.add_(1)
        self.normalized = False

    def get_space(self):
        if not self.normalized:
            sketch = self.cov_mat_sqrt.numpy()
            [_, s, Vt] = np.linalg.svd(sketch, full_matrices=False)
            self.cov_mat_sqrt = torch.from_numpy(s[:, None] * Vt)
            self.normalized = True
        curr_rank = min(self.rank.item(), self.max_rank)
        return self.cov_mat_sqrt[:curr_rank].clone() / max(1, self.num_models.item() - 1) ** 0.5
