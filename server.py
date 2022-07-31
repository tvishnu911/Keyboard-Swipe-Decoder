'''
You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.
'''

from flask import Flask, request
from flask import render_template
import time
import json
from scipy.interpolate import interp1d
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances
import sys

app = Flask(__name__)



centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])

alphaList = np.zeros((100))



template_sample_points_X = []
template_sample_points_Y = []

def generate_sample_points(pointsX, pointsY):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    
    sample_points_X, sample_points_Y = [], []
    euclideanDistances = np.sqrt(np.ediff1d(pointsX, to_begin= 0)**2 + np.ediff1d(pointsY, to_begin= 0)**2)
    prefixSums  = np.cumsum(euclideanDistances)

    prefixSums = prefixSums / prefixSums[-1]
    
    transFactorX = interp1d(prefixSums, pointsX)
    transFactorY = interp1d(prefixSums, pointsY)
    
    alphaSpace = np.linspace(0, 1, 100)
    
    sampleX = transFactorX(alphaSpace).tolist()
    sampleY = transFactorY(alphaSpace).tolist()
    
    return sampleX, sampleY
    

def normalizePoints(x, y):
    scaledCentroidX = np.mean(x, axis=1)
    scaledCentroidY = np.mean(y, axis=1)
    
    xTrans = -scaledCentroidX
    yTrans = -scaledCentroidY
    
    translation_matrix_X = np.reshape(xTrans, (-1, 1))
    translation_matrix_Y = np.reshape(yTrans, (-1, 1))
    
    return translation_matrix_X + x, translation_matrix_Y + y




def sampleTemplates(templatePointsX, templatePointsY):
    global template_sample_points_X
    global template_sample_points_Y
    for i in range(10000):
        sampleX, sampleY = generate_sample_points(templatePointsX[i], templatePointsY[i])
        template_sample_points_X.append(sampleX)
        template_sample_points_Y.append(sampleY)
        


normalized_template_sample_points_X  = []
normalized_template_sample_points_Y = []
def getDifference(li):
    max = np.max(li, axis = 1)
    min = np.min(li, axis = 1)
    return max - min
    

def preProcess(template_sample_points_X, template_sample_points_Y):
    global normalized_template_sample_points_X
    global normalized_template_sample_points_Y
    L = 250
    W = getDifference(template_sample_points_X)
    H = getDifference(template_sample_points_Y)
    
    
    # S = L / Max(W, H)
    s = L / np.max(np.array([W, H]), axis=0)
    
    
    scale = np.diag(s)
    scaled_template_points_X = np.matmul(scale, template_sample_points_X)
    scaled_template_points_Y = np.matmul(scale, template_sample_points_Y)
    

    normalized_template_sample_points_X, normalized_template_sample_points_Y = normalizePoints(scaled_template_points_X, scaled_template_points_Y)



def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider reasonable)
    to narrow down the number of valid words so that ambiguity can be avoided.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    
    
    threshold = 20
    
    
    startIdx = gesture_points_X[0]
    startJdx = gesture_points_Y[0]
    
    endIdx = gesture_points_X[len(gesture_points_X) - 1]
    endJdx = gesture_points_Y[len(gesture_points_Y) - 1]
    
    start = np.array([startIdx, startJdx])
    end = np.array([endIdx, endJdx])
    

    
    startPoints = []
    endPoints = []
    
    for idx in range(len(template_sample_points_X)):
        startPoints.append([template_sample_points_X[idx][0], template_sample_points_Y[idx][0]])
    startPoints = np.array(startPoints)
    
    for idx in range(len(template_sample_points_X)):
        endPoints.append([template_sample_points_X[idx][-1], template_sample_points_Y[idx][-1]])
    endPoints = np.array(endPoints)
    

    start = np.reshape(start, (1, -1))
    end = np.reshape(end, (1, -1))
    startDiff = euclidean_distances(start, startPoints)[0]
    endDiff = euclidean_distances(end, endPoints)[0]


    refinedIdxes = []
    
    for idx in range(len(startDiff)):
        if startDiff[idx] + endDiff[idx] < threshold:
            refinedIdxes.append(idx)
    
    refinedXes = []
    refinedYes = []
    refinedXes = np.array(template_sample_points_X)[refinedIdxes]
    refinedYes = np.array(template_sample_points_Y)[refinedIdxes]
    refinedWords = [words[idx] for idx in refinedIdxes]

    return refinedIdxes, refinedWords, refinedXes, refinedYes
    

def get_scaled_points(sample_points_X, sample_points_Y, L):
    x_maximum = max(sample_points_X)
    x_minimum = min(sample_points_X)
    W = x_maximum - x_minimum
    y_maximum = max(sample_points_Y)
    y_minimum = min(sample_points_Y)
    H = y_maximum - y_minimum
    r = L/max(H, W)

    gesture_X, gesture_Y = [], []
    for point_x, point_y in zip(sample_points_X, sample_points_Y):
        gesture_X.append(r * point_x)
        gesture_Y.append(r * point_y)

    centroid_x = (max(gesture_X) - min(gesture_X))/2
    centroid_y = (max(gesture_Y) - min(gesture_Y))/2
    scaled_X, scaled_Y = [], []
    for point_x, point_y in zip(gesture_X, gesture_Y):
        scaled_X.append(point_x - centroid_x)
        scaled_Y.append(point_y - centroid_y)
    return scaled_X, scaled_Y

def getNormalizedPoints(scaledIdxes):
    centroidDashX = np.mean(scaledIdxes[0])
    xDash = 0 - centroidDashX
    
    centroidDashY = np.mean(scaledIdxes[1])
    yDash = 0 - centroidDashY
    
    
    changeFactor = np.array([[xDash],
                                   [yDash]])
    
    normalizedPoints = changeFactor + scaledIdxes
    
    return normalizedPoints

def populateAlphas():
    global alphaList
    
    for num in range(50):
        
        alphaList[50 - num - 1], alphaList[50 + num] = num/2450, num/2450
        
def get_shape_scores(refinedIdxes, gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    # TODO: Set your own L
    
    # TODO: Calculate shape scores (10 points)
    shape_scores = []
    L = 200
    
    maxX = np.max(gesture_sample_points_X)
    minX = np.min(gesture_sample_points_X)
    
    width = maxX - minX
    
    maxY = np.max(gesture_sample_points_Y)
    minY = np.min(gesture_sample_points_Y)
    
    height = maxY - minY
    
    
    s = L / max(width, height, 1)

    originalIdxes = np.array([gesture_sample_points_X,
                                   gesture_sample_points_Y])
    
    scale = np.array([[s, 0],[0, s]])
    
    scaledIdxes = np.matmul(scale, originalIdxes)



    normalizedPoints = getNormalizedPoints(scaledIdxes)


    valid_normalized_template_sample_points_X = normalized_template_sample_points_X[refinedIdxes]
    valid_normalized_template_sample_points_Y = normalized_template_sample_points_Y[refinedIdxes]

    return getScore(normalizedPoints, valid_normalized_template_sample_points_X, valid_normalized_template_sample_points_Y)



def getScore(normalizedPoints, valid_normalized_template_sample_points_X, valid_normalized_template_sample_points_Y):
    xi = valid_normalized_template_sample_points_X 
    xj = np.reshape(normalizedPoints[0], (1, -1)) 
    xi_j = (xi - xj) ** 2
    
    
    yi = valid_normalized_template_sample_points_Y
    yj = np.reshape(normalizedPoints[1], (1, -1))
    yi_j = (yi - yj) ** 2
    
    
    
    score = np.sum((xi_j + yi_j) ** 0.5, axis=1) / 100

    return score

def get_small_d(p_X, p_Y, q_X, q_Y):
    min_distance = []
    for n in range(0, 100):
        distance = math.sqrt((p_X - q_X[n])**2 + (p_Y - q_Y[n])**2)
        min_distance.append(distance)
    return (sorted(min_distance)[0])

def get_big_d(p_X, p_Y, q_X, q_Y, r):
    final_max = 0
    for n in range(0, 100):
        local_max = 0
        distance = get_small_d(p_X[n], p_Y[n], q_X, q_Y)
        local_max = max(distance-r , 0)
        final_max += local_max
    return final_max

def get_delta(u_X, u_Y, t_X, t_Y, r, i):
    D1 = get_big_d(u_X, u_Y, t_X, t_Y, r)
    D2 = get_big_d(t_X, t_Y, u_X, u_Y, r)
    if D1 == 0 and D2 == 0:
        return 0
    else:
        return math.sqrt((u_X[i] - t_X[i])**2 + (u_Y[i] - t_Y[i])**2)

def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    
    
    location_scores = [0 for _ in range(len(valid_template_sample_points_X))]
    location_scores = np.array(location_scores)
    
    radius = 15
    # TODO: Calculate location scores (10 points)
    gestureIdxes = []
    gestureIdxesX = []
    gestureIdxesY = []
    for idx in range(100):
        gestureIdxesX.append(gesture_sample_points_X[idx])
    for idx in range(100):
        gestureIdxesY.append(gesture_sample_points_Y[idx])
    print(len(gestureIdxesX), len(gestureIdxesY))
    for idx in range(100):
        gestureIdxes.append([gestureIdxesX[idx], gestureIdxesY[idx]])
        
    for i in range(len(valid_template_sample_points_X)):
        
        templateIdxes = []
        templateIdxesX = []
        templateIdxesY = []
        # for jdx in range(num_sample_points):
        #     templateIdxesX.append(valid_template_sample_points_X[i][jdx])
        # for jdx in range(num_sample_points):
        #     templateIdxesY.append(valid_template_sample_points_Y[i][jdx])
        # print(len(gestureIdxesX), len(gestureIdxesY))
        # for jdx in range(num_sample_points):
        #     templateIdxes.append([templateIdxesX[jdx], templateIdxesY[jdx]])
        for jdx in range(100):
            xCoord = valid_template_sample_points_X[i][jdx]
            yCoord = valid_template_sample_points_Y[i][jdx]
            templateIdxes.append([xCoord, yCoord])
            
        
        
        currentDistances = euclidean_distances(gestureIdxes, templateIdxes)
        
        closestGesturePoint = np.min(currentDistances, axis=0)
        
        closestTemplatePoint = np.min(currentDistances, axis=1)
        flag = True if np.any(closestTemplatePoint > radius) or np.any(closestGesturePoint > radius) else False
        if flag:
            deltaList = np.diagonal(currentDistances)
            location_scores[i] = np.sum(np.multiply(alphaList, deltaList))
            
            
    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    
    a = 0.4
    b = 0.6
    for i in range(len(shape_scores)):
        integration_scores.append(a * shape_scores[i] + b * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
   
    
    n = 3
    

    
    if n >= len(integration_scores):
        n = len(integration_scores) - 1
    # Ref: https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    nIdxes = np.argpartition(integration_scores, n)
    
    topWords = [ valid_words[idx] for idx in nIdxes ]
    
    bestScore = integration_scores[nIdxes[0]] * (1 - probabilities[topWords[0]])
    bestWord = topWords[0]
    
    for word, idx in zip(topWords[1::], nIdxes[1::]):
        currentScore = integration_scores[idx] * (1 - probabilities[word])
        if currentScore < bestScore:
            bestScore = currentScore
            bestWord = word
        # print(bestScore, bestWord)
    return bestWord


@app.route("/")
def init():
    # sampleTemplates(template_points_X, template_points_Y)
    # preProcess(template_sample_points_X, template_sample_points_Y)
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():
    global template_sample_points_X
    global template_sample_points_Y
    global normalized_template_sample_points_X
    global normalized_template_sample_points_Y
    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    template_sample_points_X = []
    template_sample_points_Y = []
    normalized_template_sample_points_X = []
    normalized_template_sample_points_Y = []
    
    sampleTemplates(template_points_X, template_points_Y)
    preProcess(template_sample_points_X, template_sample_points_Y)
    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)
    

    refinedIndexes, valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)
    if len(valid_words) == 0:
        end_time = time.time()
        return '{"Word not found", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'

    best_word = "Word not found"
    
    shape_scores = get_shape_scores(refinedIndexes, gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'

    

if __name__ == "__main__":
    app.run()
