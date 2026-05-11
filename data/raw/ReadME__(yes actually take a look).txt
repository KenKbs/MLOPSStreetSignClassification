The dataset is taken from 'https://benchmark.ini.rub.de/gtsdb_dataset.html'
For convenience purpose the raw data available on the website is processed for easy understanding and usage!!

There are a total 900 images in the dataset, out of which 600 are allotted for training and the remaining 300 for 
testing.


In case you wondering why there are less files for labels with respect to the images, that is because not all
images contain traffic signs. So for the images with no traffic signs the text file is not present and is completely fine if you're training a YOLO model as it made to handle null images and null labels. it will ignore the images with no corresponding labels!

// ----------------- HAPPY LEARNING -----------------------  // 
