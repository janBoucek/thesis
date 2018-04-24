import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


def calculate_difference(background, frame):
    return np.clip(np.sum(cv2.absdiff(background, frame), axis=2), 0, 255)

def imshow(image):
	if len(image.shape) == 2:
		image = bw2rgb(image)
	cv2.imshow('frame', image)
	cv2.waitKey()

def crop(image):
	crop = image[0:360, 795:1500, :]
	imshow(crop)
	return crop

def crop2(image):
	crop = image[0:270, 258:810, :]
	imshow(crop)
	return crop

def bw2rgb(image):
	assert len(image.shape) == 2
	new = np.zeros(image.shape + (3,), dtype=np.uint8)
	for i in range(3):
		new[:, :, i] = image
	return new

def threshold(image, n):
	ans = np.zeros(image.shape, dtype=np.uint8)
	ans[image > n] = 255
	return bw2rgb(ans)


def create_mask(background, frame, threshold=170):
    '''
    creates the background subtraction mask
    :param background: learned background of the scene, numpy matrix [H, W, 3]
    :param frame: the processed frame, numpy matrix [H, W, 3]
    :param threshold: the threshold for the mask
    :return: the mask of the detected objects [H, W, 1], values {0, 1}
    '''

    blured_frame = cv2.blur(frame, (3, 3))

    background = background.astype(np.int16)
    blured_frame = blured_frame.astype(np.int16)
    diff = calculate_difference(background, blured_frame)


    diff_blur = diff.copy()
    diff_blur[diff < threshold/2] = 0
    diff_blur = np.clip(diff_blur, 0, 255)

    
    prior = cv2.blur(diff_blur*0.5, (60, 60))
    prior = np.clip(prior, 0, 255)
    # imshow(prior)

    diff = cv2.blur(diff, (7, 7))
    diff = diff + prior
    diff = np.clip(diff, 0, 255)

    mask = np.zeros(diff.shape, dtype=np.uint8)
    mask[diff > threshold] = 255
    return mask



def histogram():
	cap = cv2.VideoCapture("/home/emania/lepra/janboucek/omni/cut.AVI")
	SKIP = 10
	N = 100
	r, g, b = [], [], []
	# position = (292, 1150)
	position = (287, 1417)
	for i in range(N * SKIP):
		ret, frame = cap.read()
		if i % SKIP == 0:
			print(i / SKIP)
			r.append(frame[position + (2,)])
			g.append(frame[position + (1,)])
			b.append(frame[position + (0,)])
			# print(frame[position + (2,)], frame[position + (1,)], frame[position + (0,)])
			# cv2.circle(frame, (position[1], position[0]), 3, (0, 0, 255))
			# imshow(frame)

	print("red ", r)
	print("green ", g)
	print("blue ", b)

	# x = plt.hist(r, bins=25, range=(0, 255))
	red, t = np.histogram(r, bins=26, range=(0, 255))
	green, _ = np.histogram(g, bins=26, range=(0, 255))
	blue, _ = np.histogram(b, bins=26, range=(0, 255))

	red = [e / float(N) for e in red]
	green = [e / float(N) for e in green]
	blue = [e / float(N) for e in blue]

	med_r = np.mean(r)
	med_g = np.mean(g)
	med_b = np.mean(b)

	line1, = plt.plot(t[:26], red, 'r', linewidth=1, label="red channel")
	line2, = plt.plot(t[:26], green, 'g', linewidth=1, label="green channel")
	line3, = plt.plot(t[:26], blue, 'b', linewidth=1, label="blue channel")

	point1 = plt.scatter(med_r, 0, color='r')
	point2 = plt.scatter(med_g, 0, color='g')
	point3 = plt.scatter(med_b, 0, color='b')

	plt.legend([line1, line2, line3, point1, point2, point3], 
		['red channel', 'green channel', 'blue channel', 'red median', 'green median', 'blue median'])
	plt.xlabel('pixel value')
	plt.ylabel('frequency')
	plt.title('Histogram of pixel values over multiple images and medians')

	plt.grid(True)

	plt.show()


def draw_bounding_boxes(img, bb):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    for (x, y, x_max, y_max, a) in bb:
    	print((x, y, x_max, y_max, a))
        cv2.putText(img, str(int(a)), (x, y - 20), font, 0.7, color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x_max, y_max), color, 2)
    return img


def detect_bb(mask, area_treashold=1000):
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    locs = []
    for (_, c) in enumerate(cnts):
    	a = cv2.contourArea(c)
        if a > area_treashold:
            (x, y, w, h) = cv2.boundingRect(c)
            locs.append((x, y, x+w, y+h, a))
    return locs



background = cv2.imread("background_med_full.png")#[:946, :1024, :]
frame = cv2.imread("frame2.png")[:1440, :, :]


# background_frame = cv2.blur(background, (3, 3))
# blured_frame = cv2.blur(frame, (3, 3))

print("background shape ", background.shape)
print("frame shape", frame.shape)

mask = create_mask(background, frame, 220)
print(type(mask))
bbs = detect_bb(mask, 1300)
mask = bw2rgb(mask)
mask = draw_bounding_boxes(mask, bbs)

# diff = calculate_difference(background_frame, blured_frame)
# diff = calculate_difference(background, frame)
# rgb = threshold(diff, 100)

# rgb = bw2rgb(diff)
imshow(mask)



cv2.imwrite("mask.png", mask)











