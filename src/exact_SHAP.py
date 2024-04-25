import numpy as np
import scipy.integrate as integrate
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import pandas as pd
import numpy as np
import math
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def get_sigma_decomposition(conditioning_indices, Sigma):
    """
    Decomposes the covariance matrix Sigma into three submatrices based on the conditioning indices.
    cf: https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution

    Parameters:
    conditioning_indices (list): List of indices representing the conditioning variables.
    Sigma (numpy.ndarray): Covariance matrix.

    Returns:
    Sigma11 (numpy.ndarray): Submatrix of Sigma containing the non-conditioning variables.
    Sigma12 (numpy.ndarray): Submatrix of Sigma containing the cross-covariance between non-conditioning and conditioning variables.
    Sigma22 (numpy.ndarray): Submatrix of Sigma containing the conditioning variables.
    """
    non_conditioning_indices = [i for i in range(Sigma.shape[0]) if i not in conditioning_indices]
    Sigma11 = Sigma[np.ix_(non_conditioning_indices, non_conditioning_indices)]
    Sigma12 = Sigma[np.ix_(non_conditioning_indices, conditioning_indices)]
    Sigma22 = Sigma[np.ix_(conditioning_indices, conditioning_indices)]
    return Sigma11, Sigma12, Sigma22

# densities of conditional normal functions
def get_conditional_params(values, conditioning_var_inds=[1, 2], mus=np.array([0, 0, 0]), Sigma=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    """
    Calculate the expected value and variance of a conditional normal distribution.

    Parameters:
    values (list): The observed values of the conditioning variables.
    conditioning_var_inds (list): The indices of the conditioning variables.
    mus (ndarray): The mean vector of the joint normal distribution.
    Sigma (ndarray): The covariance matrix of the joint normal distribution.

    Returns:
    tuple: A tuple containing the expected value and variance of the conditional normal distribution.
    """
    
    assert(len(values) == len(conditioning_var_inds))

    # generate list of variables by length of mus
    free_var_inds = [i for i in range(len(mus)) if i not in conditioning_var_inds]

    # get Sigma decomposition
    Sigma11, Sigma12, Sigma22 = get_sigma_decomposition(conditioning_var_inds, Sigma)
    # get expected value and variance of conditional normal distribution
    exp = mus[free_var_inds] + np.dot(Sigma12, np.dot(np.linalg.inv(Sigma22), (np.array(values) - mus[conditioning_var_inds])))
    sig = Sigma11 - np.dot(Sigma12, np.dot(np.linalg.inv(Sigma22), Sigma12.T))
    return exp, sig

def normal_conditional_density(x, conditioning_values, conditioning_indices, params={"mus": np.array([0, 0, 0]), "Sigma": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])}):
    """
    Calculates the conditional density of a multivariate normal distribution.

    Parameters:
    x (array-like): The value at which to evaluate the density.
    conditioning_values (array-like): The values of the conditioning variables.
    conditioning_indices (array-like): The indices of the conditioning variables.

    Returns:
    float: The value of the conditional density at x.
    """
    exp, sig = get_conditional_params(conditioning_values, conditioning_indices, mus=params["mus"], Sigma=params["Sigma"])
    assert(len(conditioning_values) == len(conditioning_indices))
    return mvn.pdf(x, mean=exp, cov=sig)


# logit
def f_logit_3D(x, conditioning_values, conditioning_indices, coefs = np.array([1, 1, 1])):
    # select coefs using conditioning_indices and multiply with conditioning_values
    w = coefs[conditioning_indices]
    summand = np.dot(w, conditioning_values)
    # get remaining coefs
    if x is not None:
        w_remaining = coefs[[i for i in range(len(coefs)) if i not in conditioning_indices]]
        summand += np.dot(w_remaining, x)
    return np.sum(np.exp(summand)/(1 + np.exp(summand))) # sum to return scalar

def f_linear_2D(x, conditioning_values, conditioning_indices, coefs = np.array([1, 1])):
    # select coefs using conditioning_indices and multiply with conditioning_values
    w = coefs[conditioning_indices]
    summand = np.dot(w, conditioning_values)
    # get remaining coefs
    if x is not None:
        w_remaining = coefs[[i for i in range(len(coefs)) if i not in conditioning_indices]]
        summand += np.dot(w_remaining, x)
    return summand


# integrals for 3 variables
def E_f_3D(func, values, conditioning_vars, params={"mus": np.array([0, 0, 0]), "Sigma": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), "coefs": np.array([1, 1, 1])}):
    """
    Calculate the expected value of an outcome function func under different conditioning variables with normal distributions for arbitrary parameters.

    Parameters:
    values (list): List of values for the conditioning variables.
    conditioning_vars (list): List of indices of the conditioning variables.
    params (dict): Dictionary of parameters for the function func.

    Returns:
    tuple: A tuple containing the result and error of the calculation.
    """
    abs_error_tolerance = 1.49e-08
    assert(len(values) == len(conditioning_vars))
    normal_params = {"mus": params["mus"], "Sigma": params["Sigma"]}

    if len(conditioning_vars) == 3:
        result = func(x = None, conditioning_values=values, conditioning_indices=conditioning_vars)
        error = None

    if len(conditioning_vars) == 2:
        result, error = integrate.quad(lambda x: normal_conditional_density(x, conditioning_values=values, conditioning_indices=conditioning_vars, params=normal_params) * func(x, conditioning_values=values, conditioning_indices=conditioning_vars, coefs=params["coefs"]), -100, 100, epsabs=abs_error_tolerance, epsrel=abs_error_tolerance)
    
    if len(conditioning_vars) == 1:
        result, error = integrate.dblquad(lambda x,y: normal_conditional_density((x,y), conditioning_values=values, conditioning_indices=conditioning_vars, params=normal_params) * func((x,y), conditioning_values=values, conditioning_indices=conditioning_vars, coefs=params["coefs"]), -100, 100, -100, 100, epsabs=abs_error_tolerance, epsrel=abs_error_tolerance)

    if len(conditioning_vars) == 0:
        if np.allclose(params["mus"], np.array([0, 0, 0])):
            result = 0.5
            error = None
        else:
            result, error = integrate.tplquad(lambda x,y,z: mvn.pdf((x,y,z), mean=params["mus"], cov=params["Sigma"]) * func((x,y,z), conditioning_values=values, conditioning_indices=conditioning_vars, coefs=params["coefs"]), -100, 100, -100, 100, -100, 100, epsabs=abs_error_tolerance, epsrel=abs_error_tolerance)

    return result, error


def E_f_2D(func, values, conditioning_vars, params={"mus": np.array([0, 0]), "Sigma": np.array([[1, 0], [0, 1]]), "coefs": np.array([1, 1])}):
    """
    Calculate the expected value of an outcome function func under different conditioning variables with normal distributions for arbitrary parameters.

    Parameters:
    values (list): List of values for the conditioning variables.
    conditioning_vars (list): List of indices of the conditioning variables.
    params (dict): Dictionary of parameters for the function func.

    Returns:
    tuple: A tuple containing the result and error of the calculation.
    """
    abs_error_tolerance = 1.49e-08
    assert(len(values) == len(conditioning_vars))
    normal_params = {"mus": params["mus"], "Sigma": params["Sigma"]}

    if len(conditioning_vars) == 2:
        result = func(x = None, conditioning_values=values, conditioning_indices=conditioning_vars)
        error = None

    if len(conditioning_vars) == 1:
        result, error = integrate.quad(lambda x: normal_conditional_density(x, conditioning_values=values, conditioning_indices=conditioning_vars, params=normal_params) * func(x, conditioning_values=values, conditioning_indices=conditioning_vars, coefs=params["coefs"]), -100, 100, epsabs=abs_error_tolerance, epsrel=abs_error_tolerance)
    
    if len(conditioning_vars) == 0:
        if np.allclose(params["mus"], np.array([0, 0])):
            result = 0.5
            error = None
        else:
            result, error = integrate.dblquad(lambda x,y: mvn.pdf((x,y), mean=params["mus"], cov=params["Sigma"]) * func((x,y), conditioning_values=values, conditioning_indices=conditioning_vars, coefs=params["coefs"]), -100, 100, -100, 100, epsabs=abs_error_tolerance, epsrel=abs_error_tolerance)

    return result, error


class ShapleyWLS:
    def __init__(self, model_value_function, data_instance, value_function_params):
        self.M = len(data_instance)
        
        if self.M == 3:
            self.SHAP_value_function = E_f_3D
        elif self.M == 2:
            self.SHAP_value_function = E_f_2D
        else:
            raise ValueError("Only 2D and 3D data instances supported.")
        
        self.model_value_function = model_value_function
        self.data_instance = data_instance
        self.value_function_params = value_function_params
        self.Z = None
        self.W = None
        self.v = None
        self.shapley_values = None

    def build_Z(self):
        self.Z = np.ones((2**self.M, self.M+1))
        self.Z[:, 1:] = np.unpackbits(np.arange(2**self.M, dtype=np.uint8)[:, np.newaxis], axis=1)[:, -self.M:]
        print("Z built.")

    def build_W(self):
        self.W = np.zeros((2**self.M, 2**self.M))
        for i in range(2**self.M):
            sum_Z = int(np.sum(self.Z[i, 1:]))
            if sum_Z == 0 or sum_Z == self.M:
                self.W[i, i] = 10e7 # should be infinity, in practice set to large constant
            else:
                self.W[i, i] = (self.M-1)/(math.comb(self.M, sum_Z) * (self.M-sum_Z) * sum_Z)
        print("W built.")

    def build_v(self):
        self.v = np.zeros((2**self.M, 1))
        for i in tqdm.tqdm(range(2**self.M)):
            conditioning_indices = np.where(self.Z[i, 1:] == 1)[0] # indices of "playing" features
            self.v[i] = self.SHAP_value_function(func = self.model_value_function, values = self.data_instance[conditioning_indices] , conditioning_vars = conditioning_indices, params = self.value_function_params)[0]
            print(f"v_{i}: {self.v[i]}")

    def compute_shapley_values(self):
        self.build_Z()
        self.build_W()
        self.build_v()
        self.shapley_values = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(self.Z.T, self.W), self.Z)), self.Z.T), self.W), self.v)
        return self.shapley_values
    
    def get_prediction(self):
        return self.SHAP_value_function(func = self.model_value_function,values = self.data_instance, conditioning_vars = list(range(self.M)), params = self.value_function_params)[0]
    
    def plot_waterfall(self):
        assert self.shapley_values is not None; "Shapley values not computed yet."

         # get model prediction for the instance
        prediction = self.get_prediction()
        df = pd.DataFrame({"feature": ["default","x1", "x2", "x3"], "SHAP": self.shapley_values.flatten(), "value": np.insert(self.data_instance,0,0)})
        df["cumulative_SHAP"] = df["SHAP"].cumsum()

        # build waterfall plot
        f = plt.figure(figsize=(10, 5))
        sns.barplot(x="feature", y="cumulative_SHAP", data=df, color="lightblue", label="Cumulative SHAP", width = 0.2)
        sns.barplot(x="feature", y="SHAP", data=df, color = "green", width=0.8, alpha=0.5, label = "Individual SHAP")
        sns.scatterplot(x="feature", y="value", data=df, color="purple", alpha=0.6, label="Feature Value")
        ax = plt.gca()
        ax.axhline(prediction, color='blue', linewidth=2, linestyle='--', label = "Model Prediction")
        # annotate the SHAP bars with their value
        for i, v in enumerate(df["SHAP"]):
            ax.text(i, v, round(v, 2), ha='center', va='bottom')

        # add legend
        plt.legend()

        plt.show()