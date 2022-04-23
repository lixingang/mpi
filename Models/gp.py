import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch


class GaussianProcess:
    """
    The crop yield Gaussian process
    """
    def __init__(self, sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01):
        self.sigma = sigma
        self.r_loc = r_loc
        self.r_year = r_year
        self.sigma_e = sigma_e
        self.sigma_b = sigma_b

    @staticmethod
    def _normalize(x):
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_scale = np.ptp(x, axis=0, keepdims=True)

        return (x - x_mean) / x_scale

    def run(self, feat_train, feat_test, loc_train, loc_test, year_train, year_test,
            train_yield, model_weights, model_bias):

        # makes sure the features have an additional testue for the bias term
        # We call the features H since the features are used as the basis functions h(x)
        # H_train = np.concatenate((feat_train, np.ones((feat_train.shape[0], 1))), axis=1)
        # H_test = np.concatenate((feat_test, np.ones((feat_test.shape[0], 1))), axis=1)
        feat_train = torch.from_numpy(feat_train)
        feat_test = torch.from_numpy(feat_test)

        H_train = torch.cat((feat_train, torch.ones((feat_train.shape[0], 1))), dim=1)
        H_test = torch.cat((feat_test, torch.ones((feat_test.shape[0], 1))), dim=1)

        Y_train = np.expand_dims(train_yield, axis=1)
        Y_train = torch.from_numpy(Y_train)

        n_train = feat_train.shape[0]
        n_test = feat_test.shape[0]

        locations = self._normalize(np.concatenate((loc_train, loc_test), axis=0))
        years = self._normalize(np.concatenate((year_train, year_test), axis=0))
        # to calculate the se_kernel, a dim=2 array must be passed
        years = np.expand_dims(years, axis=1)

        # These are the squared exponential kernel function we'll use for the covariance
        se_loc = squareform(pdist(locations, 'euclidean')) ** 2 / (self.r_loc ** 2)
        se_year = squareform(pdist(years, 'euclidean')) ** 2 / (self.r_year ** 2)

        se_loc = torch.from_numpy(se_loc)
        se_year = torch.from_numpy(se_year)

        # make the dirac matrix we'll add onto the kernel function
        noise = torch.zeros([n_train + n_test, n_train + n_test])
        # noise[0: n_train, 0: n_train] += (self.sigma_e ** 2) * np.identity(n_train)
        noise[0: n_train, 0: n_train] += (self.sigma_e ** 2) * torch.eye(n_train)
        kernel = ((self.sigma ** 2) * torch.exp(-se_loc) * torch.exp(-se_year)) + noise
        kernel = kernel.to(torch.float32)


        # since B is diagonal, and B = self.sigma_b * np.identity(feat_train.shape[1]),
        # its easy to calculate the inverse of B
        B_inv = torch.eye(H_train.shape[1]) / self.sigma_b
        # "We choose b as the weight vector of the last layer of our deep models"
        b = torch.cat((model_weights.transpose(1, 0), torch.from_numpy(np.expand_dims(model_bias, 1))))
        K_inv = torch.linalg.inv(kernel[0: n_train, 0: n_train]).to(torch.float32)
        # The definition of beta comes from equation 2.41 in Rasmussen (2006)
        # print(H_train.dtype,K_inv.dtype ,B_inv.dtype,b.dtype,Y_train.dtype)
        beta = torch.linalg.inv(B_inv + H_train.T.mm(K_inv).mm(H_train)).mm(
            H_train.T.mm(K_inv).mm(Y_train) + B_inv.mm(b))
        # We take the mean of g(X*) as our prediction, also from equation 2.41
        pred = H_test.mm(beta) + kernel[n_train:, :n_train].mm(K_inv).mm(Y_train - H_train.mm(beta))

        return pred

    



class gp_model(GaussianProcess):
    def __init__(self, sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01):
        super(gp_model,self).__init__(sigma, r_loc, r_year, sigma_e, sigma_b)
        self.train_feat = []
        self.train_year = []
        self.train_loc = []
        self.train_y = []
        self.test_feat = []
        self.test_loc = []
        self.test_year = []
        # self.test_y = []

    def append_training_params(self, feat, year, loc, y):   
        self.train_feat.append(feat)
        self.train_year.append(year)
        self.train_loc.append(loc)
        self.train_y.append(y)

    def append_testing_params(self, feat, year, loc):
        self.test_feat.append(feat)
        self.test_year.append(year)
        self.test_loc.append(loc)


    def gp_run(self, epoch, model_weight, model_bias,):
        train_feat = np.concatenate(self.train_feat, axis=0)
        train_year = np.concatenate(self.train_year, axis=0)
        train_loc = np.concatenate(self.train_loc, axis=0)
        train_y = np.concatenate(self.train_y, axis=0)
        test_feat = np.concatenate(self.test_feat, axis=0)
        test_loc = np.concatenate(self.test_loc, axis=0)
        test_year = np.concatenate(self.test_year, axis=0)
        gp_pred = self.run(  
            train_feat, test_feat,
            train_loc, test_loc,
            train_year, test_year,
            train_y,
            model_weight, model_bias
        )

        
        # self.train_feat = []
        # self.train_year = []
        # self.train_loc = []
        # self.train_y = []
        # self.test_feat = []
        # self.test_loc = []
        # self.test_year = []
        # self.model_weight = None
        # self.model_bias = None

        return torch.squeeze(gp_pred)
    def save(self, path):
        gp_params = {
            "train_feat":self.train_feat,
            "train_year":self.train_year,
            "train_loc":self.train_loc,
            "train_y":self.train_y,
            # "test_feat":self.test_feat,
            # "test_loc":self.test_loc,
            # "test_year":self.test_year,
            # "model_weight":self.model_weight,
            # "model_bias":self.model_bias,

        }
        torch.save(gp_params,path)

    def restore(self, path):
        params = torch.load(path)
        self.train_feat = params["train_feat"]
        self.train_year = params["train_year"]
        self.train_loc = params["train_loc"]
        self.train_y = params["train_y"]
        self.test_feat = []
        self.test_loc = []
        self.test_year = []
        # self.test_y = []
         

    def clear_params(self):

        self.test_feat = []
        self.test_loc = []
        self.test_year = []
        # self.test_y = []
        self.train_feat = []
        self.train_year = []
        self.train_loc = []
        self.train_y = []

    # def clear_training_params(self):
        