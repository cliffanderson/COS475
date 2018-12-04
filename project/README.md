# Increasing Image Resolution Using Machine Learning

How to run network
- In root of repository...
- `python load_and_format_data.py`
    - This formats dataset into numpy arrays
- `python project.py`
    - This trains the network and outputs some image data
    
    
How to generate images
- In src directory...
- `javac MakeImages.java`
    - This will compile the java code
- `java MakeImages`
    - This will create three images:
        - origFullRes.png : Original 28x28 images
        - orig.png : Lower resolution 14x14 images
        - newImage.png : Prediction by the neural network using orig image