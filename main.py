from data_util import read_train_data,read_test_data,prob_to_rles,mask_to_rle,resize,np
from model import get_unet
from keras.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd

epochs = 50

# get train_data
train_img,train_mask = read_train_data()

# get test_data
test_img,test_img_sizes = read_test_data()

# get u_net model
u_net = get_unet()

# fit model on train_data
print("\nTraining...")
earlystopper = EarlyStopping(patience=5,verbose=1)
checkpointer = ModelCheckpoint('model.h5',verbose=1,save_best_only=True)
u_net.fit(train_img,train_mask,validation_split=0.1,batch_size=16,epochs=epochs,
          callbacks=[earlystopper,checkpointer])



print("Predicting")
# Predict on test data
test_mask = u_net.predict(test_img,verbose=1)

# Create list of upsampled test masks
test_mask_upsampled = []
for i in range(len(test_mask)):
    test_mask_upsampled.append(resize(np.squeeze(test_mask[i]),
                                       (test_img_sizes[i][0],test_img_sizes[i][1]), 
                                       mode='constant', preserve_range=True))

np.save('test_mask',test_mask_upsampled)

test_ids,rles = mask_to_rle(test_mask_upsampled)

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018.csv', index=False)

print("Data saved")