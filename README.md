The original code is cloned from this hf space:

https://huggingface.co/spaces/akhaliq/Molmo-7B-D-0924

The code was then modified to run on a multigpu system locally, you will need to change the file location to your model download on line 8.

I could not get fp16 or bitsandbytes quantized models to work :( ...I spent a lot of time trying

I could load the models via fp16 or bitsandbytes 4bit, but I could not inference with the model after loading.

Seems to give similar results to the online version.

![image](https://github.com/user-attachments/assets/ef344ec1-efa9-4fb0-bd57-f8adfddc7696)


![image](https://github.com/user-attachments/assets/a058854b-5b33-494c-a799-9005ea8531f2)
