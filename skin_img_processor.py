def img_processor(imgpath, meanstdpath, modelpath, weightspath):
  """
  inputs: 
        imgpath - path to image of potential skin cancer mole;
        meanstdpath - path to training mean & standard deviation;
        modelpath - path to Keras model;
        weightspath - path to weights for Keras model
  
  output: pred_dic - a prediction dictionary with diseases as keys and probabilities as values
  """
  
  from keras.models import model_from_json
  # load model
  with open(modelpath, "r") as file:
    loaded_json = file.read()
  skinmodel = model_from_json(loaded_json)

  # Load weights from file
  skinmodel.load_weights(weightspath)
  
  # Load mean and std
  train_X_mean, train_X_std = np.load(meanstdpath)
  
  # Loading, resizing image as np.array
  imagearray = np.asarray(Image.open(imgpath).resize((100,75)))
  imagearray = ((imagearray-train_X_mean)/train_X_std)
  ny, nx, nc = imagearray.shape
  imagearray = imagearray.reshape(1 ,ny, nx, nc)
                
  pred_vec = skinmodel.predict(imagearray).flatten()
  
  pred_dict = {'Actinic keratoses' : pred_vec[0], 'Basal cell carcinoma' : pred_vec[1],
                   'Benign keratosis-like lesions' : pred_vec[2], 'Dermatofibroma' : pred_vec[3],
                   'Melanocytic nevi' : pred_vec[4], 'Melanoma' : pred_vec[5], 'Vascular lesions' : pred_vec[6]
              }
                                      
  
  return pred_dict
  