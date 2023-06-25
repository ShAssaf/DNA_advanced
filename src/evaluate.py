import os

import torch
from pytorch_lightning import loggers


from src.netmodel import Net

# # Assuming you have a file named 'model.pth' containing the saved model
# model =  Net(input_shape=[6,100], num_classes=10) # Create an instance of your model class
# model.load_state_dict(torch.load('./model/kmer_model.pth'))
# model.eval()  # Set the model to evaluation mode if needed
# Step 1: Preprocess the new sample

# Step 2: Prepare the input tensor
# input_tensor = torch.tensor(new_sample)  # Convert the sample to a PyTorch tensor
#
# # Step 3: Perform inference
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():  # Disable gradient computation for inference
#     output = model(input_tensor)  # Pass the input tensor through the model

# Step 4: Interpret the predictions
# Here, you can extract the predicted class label, probabilities, or perform any other post-processing


from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
tb_logger = loggers.TensorBoardLogger('logs/')

# Retrieve event file from logger
event_file = tb_logger.experiment.get_logdir()
print(next(os.walk(event_file + '/.')))

#event_file += '/' + next(os.walk(event_file + '/.'))[1][0] + '/events.out.tfevents.' + next(os.walk(event_file + '/.'))[2][0][20:]

# Create an event accumulator
event_acc = EventAccumulator(event_file)
event_acc.Reload()

# Get the scalar events
training_loss = [s.value for s in event_acc.Scalars('train_loss')]
val_loss = [s.value for s in event_acc.Scalars('val_loss')]

# Plot the losses
plt.plot(training_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.show()