### HW09 answers
How long does it take to complete the training run? (hint: this session is on distributed training, so it will take a while)  
I have been running for 72 hrs and it was only at 8000 step.  

Do you think your model is fully trained? How can you tell?  
not really as I couldn't train for long to see the loss being flattened.  

Were you overfitting?  
I don't think the model was overfitting since the train loss and evaluation loss are approximately at the same rate and the training data is enough. Also I didn't train for minimum steps.   

Were your GPUs fully utilized?  
Yes, I could see gpu's being utilized using nvidia-smi command  

Did you monitor network traffic (hint: apt install nmon ) ? Was network the bottleneck?  
In my case network was bottleneck as I did not get requested speed of (1gbps)  

Take a look at the plot of the learning rate and then check the config file. Can you explan this setting?  
The learning rate was based on the transformer policy.  

How big was your training set (mb)? How many training lines did it contain?  
train.clean.en.shuffled.BPE_common.32K.tok – 959MB – 4524869 lines  
train.clean.de.shuffled.BPE_common.32K.tok – 1023MB – 4524869 lines  

What are the files that a TF checkpoint is comprised of?  
It consists of the checkpoint, the model data, index and meta files  
checkpoint  
events.out.tfevents.timestamp.hostname  
graph.pbtxt  
model.ckpt-0.data-00000-of-00001  
model.ckpt-0.index  
model.ckpt-0.meta  

How big is your resulting model checkpoint (mb)?  
The resulting model checkpoint was 812MB  
 
Remember the definition of a "step". How long did an average step take?  
30 seconds per step  

How does that correlate with the observed network utilization between nodes?  
if there is a bottleneck in the network then the time taken by a step will be more, in my case I saw that the network utilization between nodes was a problem and hence the time taken for each step was high around 30 secs.  
