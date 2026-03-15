from convolutionGS import *

def main():
    image = "image.jpg"
    kernel = embossing_3x3
    padding = "no_padding"

    convolution_GS(image, kernel, padding)

if __name__ == "__main__":
    main()
