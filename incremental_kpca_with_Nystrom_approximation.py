# -*- coding: utf-8 -*-

from __future__ import division, print_function

from kernels import kernel_matrix, adjust_K
from eigen_update import expand_eigensystem, update_eigensystem

from copy import copy, deepcopy

import numpy as np
from numpy import dot, diag, ones
from scipy import linalg


def kernel_error(*args, **kwargs):
    raise ValueError("Kernel function not specified!")


class IncrKPCA_real(object):
    """
    Incremental kernel PCA

    Parameters
    ----------
    X : numpy.ndarray, 2d
        Data matrix
    mmax : int
        Maximum size of kernel matrix (or Nyström subset)
    kernel : callable
        Kernel function
    adjust : bool
        Whether to adjust the mean
    nystrom : bool
        Calculate incremental Nyström approximation instead
    maxiter : int
        Maximum number of iterations

    Yields
    ------
    The iteration number, the calculated eigenvectors and eigenvalues, and in
    the case of the Nyström method, also the approximate eigenvalues and
    eigenvectors

    """
    #这里m0=800,表示从第800个SWAG收集的dev_vector开始计算L和U；
    #mmax=200表示最多新收集的dev_vector个数
    def __init__(self,X, m0=800,mmax=200, kernel=kernel_error, adjust=False,
                     nystrom=False, maxiter=500): #maxiter其实没有多大用，就是用来计数，j-mmax表示Omega计算得到
        #L_slide返回r=1的个数即没有找到L_slide的次数；

        # Setup default arguments
        self.X = X

        n = X.shape[0]  # n代表样本个数，行数
        self.idx = np.arange(n)  # X=dev_vector按顺序保存,而作者原代码在这里将索引打乱了（意味着当我输入deviation vector进来是，唯一的亮点）
        self.i = self.idx[n-1] #表示最后一个元素的索引

        self.m0 = m0 # m0=20=self.i
        self.j = 0
        self.n = n
        self.maxiter = maxiter
        self.mmax = mmax
        # if mmax is None:
        #     mmax = n
        # self.mmax = min(mmax, n) #最大核矩阵

        # self.idx = np.random.permutation(n) #np.random.permutation(10)= array([8, 2, 7, 1, 3, 5, 6, 9, 0, 4])
        self.kernel = kernel
        self.adjust = adjust
        self.nystrom = nystrom

        # Initial eigensystem

        cols = self.idx[:n-1] #X.shape[0]=n,则X[n-1]是最后一个新增的样本X[0~n-2]表示前面n-1=800个初始样本
        K_mm = kernel_matrix(X, kernel, cols, cols) #表示对X数组按行索引值cols求kernel如
        # kernel(X[cols[0]],X[cols[1]])=kernel(X[8],X[2])... 即对X的行子集进行求kernel

        if self.adjust:
            self.L, self.U, self.capsig, self.K1 = init_vars(K_mm) #capsig为K_mm中的所有元素的和，K1为每行和
        else:
            self.L, self.U = linalg.eigh(K_mm)
            #Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.

        if self.nystrom:
            self.K_nm = kernel_matrix(X, kernel, range(n), cols) #n=X.shape[0]样本个数；cols=idx[:X.shape[0]-1]
            #表示将X[0..n]与X[idx[cols]]添加在一起初始化，

    def __next__(self):
        """
        Initiate the next iteration

        """
        if self.i == self.X.shape[0]: #表示如果索引i=n，那么就超出了对新增一个样本的处理，则退出
            # print("stopIteration self.i=", self.i)
            # print("self.j=",self.j)
            raise StopIteration

        # self.idx = np.random.permutation(n) ，len(self.idx)=n; self.m0=20
        if self.j == self.n-self.m0 or self.j == self.maxiter: #self.j = 0; self.maxiter=500;
            # print("StopIteraton self.j=", self.j)
            # print("self.i=",self.i)
            # print("len(self.idx)=",len(self.idx))
            raise StopIteration

        if not self.adjust:#self.adjust=False
            rc = self.update_eig() #对新加入的索引为i=20的样本X[20]进行更新
            #这句话里面调用self.update_K_nm()，通过nystrom近似从而更新得到新的K_nm
        else:# self.adjust=True
            rc = self.update_eig_adjust()

        out = (self.i-1, self.L, self.U) #i-1是表示迭代索引i先+1，再-1；
        if self.nystrom:
            if not rc: #rc=0 ,继续运行下去，满足倒数第二行rc=0的要求即之间输出out,
                #上面这句话的意思是如果要nystrom近似，并且rc=0则不需要__next__迭代了，直接out;
                # 如果是nystrom近似，但rc=1,则不进行nystrom_appromixation(),继续next迭代；
                self.L_nys, self.U_nys = nystrom_approximation(
                        self.L, self.U, self.K_nm)
                out = (self.i-1, self.L, self.U, self.L_nys, self.U_nys)

        self.j += 1 # j控制着迭代次数

        if rc: #rc=1继续下一个迭代
            return self.__next__()
        else: #rc=0 则输出out=(i-1, L, U)
            return out

    def update_K_nm(self):
        """
        Update K_nm for one iteration by adding another column

        """
        i = self.i
        K_ni = kernel_matrix(self.X, self.kernel, range(self.n), [self.idx[i]])
        self.K_nm = np.c_[self.K_nm, K_ni] #np.c_表示按列将K_ni添加进矩阵K_nm中，

    def update_eig(self):
        """
        Update the eigen decomposition of K with an additional data point

        """
        #初始化时 self.i = self.idx[n-1] 表示最后一个样本的索引
        i = self.i # index of new data points / size of existing K
        col = self.idx[i]#新数据的索引,self.i = m0 =20表示新增的一列索引从
        # idx[20]取数据X(idx[20])，因为前0-19为初始化的
        cols = self.idx[:i+1] #cols=idx[:21]增加一个索引值，索引值为0...i=20之间的值；
        sigma, k1, k0 = create_update_terms(self.X, cols, col, self.kernel)

        #在初始化阶段K_mm分解为L和U，这里添加
        L, U = expand_eigensystem(self.L, self.U, k0[-1][0]) #添加新的特征值k0[-1][0]到现有的特征值L和特征向量U,返回新的L和U

        # Order the eigenpairs and update terms
        idx = np.argsort(L) #对一维数组L进行索引所对应的特征值进行从小到大排序，返回索引值
        L = L[idx]   #L为特征值且L为从小到大的排序
        U = U[:,idx] #U为特征向量,表示列按索引值（对应从小到大的特征值）从左到右依次调整顺序
        U = U[idx,:] #在调整完了列后，同样原理按行调整顺序得到更新后的特征向量
        k1 = k1[idx,:] #sigma=4/k,k为一个值；k1=kernel_matrix且最后一行用k/2替代；
        k0 = k0[idx,:] #k0=k1且最后一行为k/4

        L, U = update_eigensystem(L, U, k1, sigma)
        #这一行sigma和下一行-sigma对应着论文中的rankoneupdate(sigma,k1,L,U)函数
        if isinstance(L, np.ndarray):
            L, U = update_eigensystem(L, U, k0, -sigma)

        if isinstance(L, np.ndarray):
            #只要L是np.ndarray的实例化，那么rc=0，那么就在__next__()中return out,而不需要
            #rc=1时self.__next__()，分析一下L是在每次的update_eigensystem()后是np.ndarray的实例化吗？
            if self.nystrom:
                self.update_K_nm() #这句话的作用是求K_ni=kernel_matrix(X,kernel,range(n),[idx[i]])
                #即将第i个索引对应的X[idx[i]]在需要nystrom近似时更新K_nm=np.c_[K_nm,K_ni]
            self.idx[:i+1] = self.idx[:i+1][idx] # reorder index; i表示一个新的数据点的
            # 注意[idx]这里的idx=np.argsort(L)，即对self.idx[:i+1]进行从小到大重新排序后的索引
            #如idx=array([2,1,0,3,4])表示从小到大的某个数组的索引,
            #而idx[:5][idx]=array([0,1,2,3,4])，这里idx[:5]就代表要重新排序的数组，[idx]即为数组idx[:5]从小到大的新序列
            if self.nystrom:
                self.K_nm = self.K_nm[:,idx] # Reorder columns，这个idx=np.argsort(L)
                #这是计算nystrom近似时K_nm唯一与L，U有关系的地方，K_nm列的重新排序要靠
                # L,U=expand_eigensystem(L,U,k0)返回的L，而这个L是在加入新的k0后才重新排列，其实也没有到更新L，U的那个阶段；
                #所以结论是nystrom是一个独立的计算近似L_nys，U_nys的方法，而与本文其他更新的L，U没关系
                #但在__next__()中最后的部分L_nys,U_nys=nystrom_approximation(L,U,K_nm)却又用到更新后的L，U，所以nystrom近似
                #是最复杂的存在，前面的incremental_experiment实验只需要更新后的L，U即可，
                # 而nystrom在更新后的L，U的基础上继续多了一次操作
            self.i, self.L, self.U = i+1, L, U #i+1表示新增加一个数据样本
        #     rc = 0
        # else: # Ignore data example,即将第i索引去除，也就是去除了索引号为i的样本
        #     self.idx[i:-1] = self.idx[i+1:]#表示从i+1到最后一个值赋值给idx从i到倒数第二个位置
        #     self.idx = self.idx[:-1] #idx[:-1]表示除了最后一个索引值，其他都赋值给idx，即最后一个值舍弃
        #     rc = 1
        #
        # return rc

    def update_eig_adjust(self):
        """
        Update the kernel PCA solution including adjustment of the mean.

        """
        i = self.i #self.i表示最后一个元素的索引，self.idx=np.arnage(10); self.i=self.idx[9]=9
        col = self.idx[i] #i为idx最后一个值的索引，所以idx[i]表示取idx最后一个值；如果idx[:i]表示只能取idx的索引为0...i-1的值
        cols = self.idx[:i+1] #idx[:i+1]表示取idx中索引为0...i的值
        k = kernel_matrix(self.X, self.kernel, cols, [col]) # OK
        a = k[:-1,:] #除k的最后一行赋值给a
        a_sum = np.sum(a) #对a中的所有元素求和
        k_sum = np.sum(k) #对k中的所有元素求和
        capsig2 = self.capsig + 2 * a_sum + k[-1,0] #capsig为K_mm所有元素的和
        C = -self.capsig/i**2 + capsig2/(i+1)**2
        u =  self.K1/(i*(i+1)) - a/(i+1) + 0.5 * C * ones((i,1))
        u1 = 1 + u
        u2 = 1 - u
        sigma_u = 0.5

        K1 = np.r_[self.K1 + a, [[k_sum]]]
        capsig = capsig2
        v = k - (ones((i+1,1)) * k_sum + K1 - capsig/(i+1)) / (i+1)
        v1 = deepcopy(v)
        v2 = deepcopy(v)
        v0 = copy(v[-1,0])
        v1[-1,0] = v0 / 2
        v2[-1,0] = v0 / 4
        sigma_k = 4 / v0

        # Apply rank one updates
        L, U = update_eigensystem(self.L, self.U, u1, sigma_u)
        if isinstance(L, np.ndarray):
            L, U = update_eigensystem(L, U, u2, -sigma_u)
        if isinstance(L, np.ndarray):
            L, U = expand_eigensystem(L, U, v0/4)

            # Ordering
            idx = np.argsort(L)
            L = L[idx]
            U = U[:,idx] #U的列按照索引idx取出
            U = U[idx,:] #在取出特定列后的U的基础上取出idx的U的行，从而使U=U[idx,idx]
            v1 = v1[idx,:]
            v2 = v2[idx,:]

            L, U = update_eigensystem(L, U, v1, sigma_k)

        if isinstance(L, np.ndarray):
            L, U = update_eigensystem(L, U, v2, -sigma_k)

        if isinstance(L, np.ndarray):
            #f self.nystrom:
            #    self.update_K_nm()
            K1 = K1[idx,:]
            self.idx[:i+1] = self.idx[:i+1][idx] # Reorder index
            if self.nystrom:
                self.K_nm = self.K_nm[:,idx] # Reorder columns
            self.i, self.L, self.U, self.K1 = i+1, L, U, K1
            self.capsig = capsig
        #     rc = 0
        # else: # Ignore data example
        #     self.idx[i:-1] = self.idx[i+1:]
        #     self.idx = self.idx[:-1]
        #     rc = 1
        #
        # return rc

    def get_idx_array(self):
        return self.idx[:self.mmax]

    def __iter__(self):
        return self

def init_vars(K_mm):
    """
    Create initial eigenpairs and adjustment variables

    Parameters
    ----------
    K_mm : np.ndarray, 2d
        Initial kernel matrix

    Returns
    -------
    Initial eigenvalues L and eigenvectors U, sum of all values of K_mm
    (capsig), sum of the rows of K_mm (K1)

    """
    m0 = K_mm.shape[0]
    Kp = adjust_K(K_mm)
    L, U = linalg.eigh(Kp)
    capsig = np.sum(np.sum(K_mm))
    K1 = dot(K_mm, ones((m0, 1)))

    return L, U, capsig, K1

def create_update_terms(X, cols, col, kernel):
    """
    Create the terms supplied to eigenvalue update algorithm

    Parameters
    ----------
    X : np.ndarray, 2d
        Data matrix
    cols : np.ndarray, 1d
        Indices of columns to create the kernel matrix
    col : float
        The additional column index

    Returns
    -------
    Parameters supplied to update algorithm for
    eigen decomposition

    """
    #col=self.idx[20]表示取X第20个索引对应的行与cols中对应的行索引号所对应的行进行kernel计算
    # k1.shape=(len(cols),1)
    k1 = kernel_matrix(X, kernel, cols, [col]) #col=self.idx[i]=self.idx[20],col表示第20个索引所对应的索引值
    k = copy(k1[-1][0]) # k1[-1]表示最后的一行，k1[-1][0]表示最后一行的第一个数
    k1[-1] = k / 2      # k1[-1]表示最后一行用最后一行的第一个数的一半来填充

    k0 = deepcopy(k1) # numpy pass by reference
    k0[-1] = k / 4

    sigma = 4 / k

    return sigma, k1, k0 #k1核的最后一行的值用k/2替代，而k0用k/4替代

def nystrom_approximation(L, U, K_nm):
    """
    Create the Nyström approximations to the eigenpairs of the kernel matrix

    Parameters
    ----------
    U : numpy.ndarray, 2d
        eigenvector matrix for the matrix K_mm
    L : numpy.ndarray, 1d
        eigenvalues for the matrix K_mm
    K_nm : numpy.ndarray, 2d
        the m sampled columns of K

    Returns
    -------
    Nyström approximate eigenvalues L and eigenvectors U

    """
    #K_nm：从核矩阵K中采样m列得到的
    n, m = K_nm.shape #n代表行数-特征数，m代表列数-样本数
    L_nys = n/m * L #特征值的近似值：行数/列数 * L
    U_nys = np.sqrt(m/n) * dot(K_nm, dot(U, diag(1/L)))
    #L是特征值，np.diag(L)表示取主对角元素
    #k0=array([[0, 1, 2, 3],
       # [4, 5, 6, 7],
       # [4, 4, 4, 4]])  #np.diag(k0)=array([0,5,4])
    #np.dot(k0.transpose,np.diag(k0))=array([36,41,46,51])

    return L_nys, U_nys

