from PIL import Image
import requests

def read_images():
    with open("./algorithms_examples/dataset/chrismas_images.txt", "r") as image_urls:
        print("Fails to read :\n")
        i = 1
        for url in image_urls:
            try:
                image = Image.open(requests.get(url, stream=True).raw)
                image.save("./dataset/chrismas/image{}.jpg".format(i))
                pass
                i += 1
            except:
                print(url)

if __name__ == "__main__":
    read_images()