
import skimage.draw
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import imutils
"""from imutils import face_utils"""
import numpy as np
"""import collections"""
import cv2
from collections import Counter
from sklearn.cluster import KMeans
import dlib


def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def captureimage(request):
    return render(request, 'captureimage.html')

def contact(request):
    return render(request, 'contact.html')

def disclaimer(request):
    return render(request, 'disclaimer.html')

def knowmore(request):
    return render(request, 'knowmore.html')

def detectedSkin(request):
    return render(request, 'detectedSkin.html')

def thanku(request):
    return render(request, 'thanku.html')

def extractSkin(image):
    img = image.copy()
    black_img = np.zeros((img.shape[0],img.shape[1],img.shape[2]),dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)
    skin = cv2.bitwise_and(img, img, mask=skinMask)
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):
    hasBlack = False
    occurance_counter = Counter(estimator_labels)
    def compare(x, y): return Counter(x) == Counter(y)
    for x in occurance_counter.most_common(len(estimator_cluster)):
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]
        if compare(color, [0, 0, 0]) == True:
            del occurance_counter[x[0]]
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break
    return (occurance_counter, estimator_cluster, hasBlack)

def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):
    occurance_counter = None
    colorInformation = []
    hasBlack = False
    if hasThresholding == True:
        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black
    else:
        occurance_counter = Counter(estimator_labels)
    totalOccurance = sum(occurance_counter.values())
    for x in occurance_counter.most_common(len(estimator_cluster)):
        index = (int(x[0]))
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index
        color = estimator_cluster[index].tolist()
        color_percentage = (x[1]/totalOccurance)
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}
        colorInformation.append(colorInfo)
    return colorInformation

def extractDominantColor(image, number_of_colors=1, hasThresholding=False):
    if hasThresholding == True:
        number_of_colors += 1
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0]*img.shape[1]), 3)
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)
    estimator.fit(img)
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation

def plotColorBar(colorInformation):
    color_bar = np.zeros((100, 500, 3), dtype="uint8")
    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])
        color = tuple(map(int, (x['color'])))
        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar


def result(request):
    if request.method == 'POST' and 'Choose file' in request.FILES:

        rgb_lower = [45, 34, 30]
        rgb_higher = [255, 219, 172]

        skin_shades = {
            'dark': [rgb_lower, [170, 100, 65]],
            'mild': [[170, 100, 65], [200, 140, 100]],
            'fair': [[200, 140, 100], rgb_higher]
        }
        convert_skintones = {}
        for shade in skin_shades:
            convert_skintones.update({
                shade: [
                    (skin_shades[shade][0][0] * 256 * 256) + (skin_shades[shade][0][1] * 256) + skin_shades[shade][0][
                        2],
                    (skin_shades[shade][1][0] * 256 * 256) + (skin_shades[shade][1][1] * 256) + skin_shades[shade][1][2]
                ]
            })

        doc = request.FILES
        doc_name = doc['Choose file']
        fs = FileSystemStorage()
        file_path = fs.save(doc_name.name, doc_name)
        file_path = fs.url(file_path)

        file_path = file_path[1:]

        img = cv2.imread(file_path)

        original_img = imutils.resize(img, width=250)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        rect = detector(original_img)[0]
        sp = predictor(original_img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])

        outline = landmarks[[*range(17), *range(26, 16, -1)]]

        Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])

        image = np.zeros(original_img.shape, dtype=np.uint8)
        image[Y, X] = original_img[Y, X]

        skin = extractSkin(image)


        unprocessed_dominant = extractDominantColor(skin, number_of_colors=1, hasThresholding=True)

        decimal_lower = (rgb_lower[0] * 256 * 256) + (rgb_lower[1] * 256) + rgb_lower[2]
        decimal_higher = (rgb_higher[0] * 256 * 256) + (rgb_higher[1] * 256) + rgb_higher[2]
        dominantColors = []
        for clr in unprocessed_dominant:
            clr_decimal = int((clr['color'][0] * 256 * 256) + (clr['color'][1] * 256) + clr['color'][2])
            if clr_decimal in range(decimal_lower, decimal_higher + 1):
                clr['decimal_color'] = clr_decimal
                dominantColors.append(clr)
        skin_tones = []
        if len(dominantColors) == 0:
            skin_tones.append('Unrecognized')
        else:
            for color in dominantColors:
                for shade in convert_skintones:
                    if color['decimal_color'] in range(convert_skintones[shade][0], convert_skintones[shade][1] + 1):
                        skin_tones.append(shade)



        """image = Image.fromarray(data)
        print(image)"""
        var = "Your Skintone is " + skin_tones[0]

        return render(request, "detectedSkin.html",{"Result1":var})
    else:
        var = "NO IMAGE UPLOADED !!!"
        return render(request, "detectedSkin.html",{"Result1":var})
