# Increasing Image Resolution Using Machine Learning

How to run network:
~~~~~~~~~~~~~~~~~~~
- In root of repository, run `python project.py`

	- Loads and formats the MNIST handwritten digit dataset

    - Loads/generates an algorithmically compressed version of the dataset

    - Trains the neural network using these datasets

    - Predicts the high resolution images for the test set

    - Displays 10 of the predictions, alongside their corresponding original
    	and compressed images from the test set

    - Writes the image data to a text file
    
    - Generate three images:
        - comp.png : Compressed 14x14 image
        - origFullRes.png : Original 28x28 image
        - newImage.png : Prediction by the neural network based on compressed image


