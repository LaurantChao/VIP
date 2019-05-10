### VIP test mode
mode = 1
print("Testing model")
    
for split in range(0,10,1):
    x = Data[:,0:size_feature]
    y = Data[:,size_feature]
    std_y = y.std()
    y = preprocessing.scale(y) # standardization
    y = y.reshape(y.shape[0],1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=38+split)
    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    # preprocess data
    mean_x = (x_train.mean(1)).reshape(x_train.shape[0],1)
    std_x = x_train.std(1).reshape(x_train.shape[0],1)
    x_train = (x_train - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x
#    

    
    # use best hyper parameters from validation
    init = [best_grid[split,0],1/best_grid[split,1]] 
    # train the model on the full training set (during validation phase, the model is only trained on part of the training set, 
    # therefore we train the VIP on the full training set under the best hyper parameters from validation
    structure = [10,10,1] # BNN structure
    exec(open("VIP_fun_speedup_recon_alpha_classic_validate.py").read())
    
    
    
    llh_VIP_agg[split] = llh_test_VIP-np.log(std_y)
    rmse_VIP_agg[split] = rmse_test_VIP*std_y
    print("Train and test on split {:.0f} completed.".format(split)) 
    print("Test llh of this split: {:.4f}.".format(llh_test_VIP-np.log(std_y)))
    print("Test rmse of this split: {:.4f}.".format(rmse_test_VIP*std_y))



print("negative log likelihood averaged on all split {:.4f} + {:.4f}. RMSE averaged on all split {:.4f}+{:.4f}.".format(-llh_VIP_agg.mean(), llh_VIP_agg.std()/np.sqrt(n_split),rmse_VIP_agg.mean(),rmse_VIP_agg.std()/np.sqrt(n_split)))
