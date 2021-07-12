import copy

from openvino.inference_engine import IECore
import cv2

from pose_extractor import extract_poses

face_model   = 'face-detection-0200'
person_model = 'person-detection-0200'
age_model    = 'age-gender-recognition-retail-0013'
hp_model     = 'human-pose-estimation-0001'

face_device   = 'CPU'
person_device = 'CPU'
age_device    = 'CPU'
hp_device     = 'CPU'


class openvino_model:
    def __init__(self):
        pass

    def model_load(self, ie, model_name, device='CPU', verbose=False):
        model_tmpl = './intel/{0}/FP16/{0}.{1}'
        self.net    = ie.read_network(model_tmpl.format(model_name, 'xml'))
        self.exenet = ie.load_network(self.net, device)
        self.iblob_name  = list(self.exenet.input_info)
        self.iblob_shape = [ self.exenet.input_info[n].tensor_desc.dims for n in self.iblob_name]
        self.oblob_name  = list(self.exenet.outputs)
        self.oblob_shape = [ self.exenet.outputs[n].shape for n in self.oblob_name]
        self.device = device
        if verbose:
            print(model_name, self.iblob_name, self.iblob_shape, self.oblob_name, self.oblob_shape)

    def image_infer(self, *args):
        inputs = {}
        for img, bname, bshape in zip(args, self.iblob_name, self.iblob_shape):
            n,c,h,w = bshape
            img = cv2.resize(img, (w,h))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OMZ models expect BGR image input
            img = img.transpose((2,0,1))
            img = img.reshape((n,c,h,w))
            inputs[bname] = img
        self.res = self.exenet.infer(inputs)
        return self.res


def draw_bounding_boxes(img, bboxes, threshold=0.8, color=(255,255,255), thickness=2):
    h, w, c = img.shape
    images = []
    for bbox in bboxes:
        id, label, conf, xmin, ymin, xmax, ymax = bbox
        if conf > threshold:
            x1 = max(0, int(xmin * w))
            y1 = max(0, int(ymin * h))
            x2 = max(0, int(xmax * w))
            y2 = max(0, int(ymax * h))
            image = img[y1:y2, x1:x2]
            images.append([image, [label, conf, (x1, y1), (x2, y2)]])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)          
    return images



def renderPeople(img, people, scaleFactor=4, threshold=0.5):
    limbIds = [
            [ 1,  2], [ 1,  5], [ 2,  3], [ 3,  4], [ 5,  6], [ 6,  7], [ 1,  8], [ 8,  9], [ 9, 10], [ 1, 11],
            [11, 12], [12, 13], [ 1,  0], [ 0, 14], [14, 16], [ 0, 15], [15, 17], [ 2, 16], [ 5, 17] ]

    limbColors = [
        (255,  0,  0), (255, 85,  0), (255,170,  0),
        (255,255,  0), (170,255,  0), ( 85,255,  0),
        (  0,255,  0), (  0,255, 85), (  0,255,170),
        (  0,255,255), (  0,170,255), (  0, 85,255),
        (  0,  0,255), ( 85,  0,255), (170,  0,255),
        (255,  0,255), (255,  0,170), (255,  0, 85)
    ]
    # 57x32 = resolution of HM and PAF
    scalex = img.shape[1]/(57 * scaleFactor)
    scaley = img.shape[0]/(32 * scaleFactor)
    for person in people:
        for i, limbId in enumerate(limbIds[:-2]):
            x1, y1, conf1 = person[limbId[0]*3:limbId[0]*3+2 +1]
            x2, y2, conf2 = person[limbId[1]*3:limbId[1]*3+2 +1]
            if conf1>threshold and conf2>threshold:
                cv2.line(img, (int(x1*scalex),int(y1*scaley)), (int(x2*scalex),int(y2*scaley)), limbColors[i], 2)


def main():
    ie = IECore()
    face_det = openvino_model()
    face_det.model_load(ie, face_model, face_device, True)

    age_det = openvino_model()
    age_det.model_load(ie, age_model, age_device, True)

    person_det = openvino_model()
    person_det.model_load(ie, person_model, person_device, True)

    hp_det = openvino_model()
    hp_det.model_load(ie, hp_model, hp_device, True)

    video_source ='../../run_demos/sample-videos/face-demographics-walking.mp4'
    #video_source = '../../run_demos/sample-videos/people-detection.mp4'
    #video_source = 0  # webCam
    cap = cv2.VideoCapture(video_source)

    key = -1
    while key != 27:
        _, input_img = cap.read()
        img = copy.deepcopy(input_img)

        # Detect faces and draw bounding boxes
        res = face_det.image_infer(input_img)
        faces = draw_bounding_boxes(img, res['detection_out'][0][0])

        # Estimate gender and age for each face
        for face, attr in faces:
            res = age_det.image_infer(face)
            age = int(res['age_conv3'].flatten() * 100)
            gen = res['prob'].flatten()
            th = 0.6
            if gen[0] > th:
                gender = 'F'
                color = (64, 0, 255)
            elif gen[1] > th:
                gender = 'M'
                color = (255, 64, 0)
            else:
                gender = 'U'
                color = (0, 0, 0)
            msg = '{} {}'.format(age, gender)
            cv2.putText(img, msg, attr[2], cv2.FONT_HERSHEY_PLAIN, 2, color, 2, cv2.LINE_AA)

        # Detect human body and draw bounding boxes
        res = person_det.image_infer(input_img)
        people = draw_bounding_boxes(img, res['detection_out'][0][0])

        # Estimate pose of people
        res = hp_det.image_infer(input_img)
        heatmaps = res['Mconv7_stage2_L2'][0]
        PAFs     = res['Mconv7_stage2_L1'][0]
        poses = extract_poses(heatmaps[:-1], PAFs, 4)                      # Construct poses from HMs and PAFs
        renderPeople(img, poses, 4, 0.2)

        # Display number of people
        msg = 'people count = {}, pose count = {}'.format(len(people), len(poses))
        cv2.putText(img, msg, (0, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_AA)

        # Show image and get a key
        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key == ord(' '):
            while cv2.waitKey(30) != ord(' '): pass

if __name__ == '__main__':
    main()

'''
face-detection-0200
face-detection-0202
face-detection-0204
face-detection-0205
face-detection-0206
face-detection-adas-0001
face-detection-retail-0004
face-detection-retail-0005

person-detection-0106
person-detection-0200
person-detection-0201
person-detection-0202
person-detection-0203
person-detection-retail-0002
person-detection-retail-0013

age-gender-recognition-retail-0013

human-pose-estimation-0001
'''
