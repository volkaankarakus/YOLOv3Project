# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 22:01:56 2021

@author: VolkanKarakuş
"""

import cv2
import numpy as np

img=cv2.imread("D:/YOLO/yoloPretrainedImage/image.jpg")
scalePercent=60
w=int(img.shape[1]*scalePercent/100)
h=int(img.shape[0]*scalePercent/100)
dim=(w,h)
img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)


# en ve boy icin console ekranina img.shape yaz.
img_width=img.shape[1]
img_height=img.shape[0]

#blob 4 boyutludur: 5 parametresi:image,resmin yeniden boyutlandirilmasi icin scale factor(en optimal 1/255),
#blobun kaca kaclik oldugu(416lik yoloyu indirmistik),bgr'dan rgb'ye cevir
#crop (kirpilma istemiyorum.)
imgblob=cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)

#labellari girelim. Taniyacagi nesneler
labels=["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
        "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
        "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
        "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
        "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
        "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

colors =["0,255,255","0,0,255","255,0,0","120,150,160","50,40,20","90,0,70"]

# herbir elemani tektek dolasicam ve int degerine ceviricem.
#color degeri bunlari tektek kendine alicak,
#bunlari , kisimlarindan ayirip herbir sayiyi int'e cevircem. Cunku suan string turunde.
colors= [np.array(color.split(",")).astype("int") for color in colors] 

#buraya kadar colors int oldu. Ama biz bunun tek bir array icine almak istiyoruz.
colors=np.array(colors)

#ben bu matris icinden rastgele degerler secicem.
#ama diyelim ki cok fazla nesnem var. O zaman bu matrisi alt alta defalarca eklemem lazim.
colors=np.tile(colors,(18,1)) # 18 tane altina ekledik. 1 koyarak yanina birsey eklenmedi. Ayni matris kaldi.

#%% THIRTH PART

#conf ve weights filelari model icinde tutucaz.
model=cv2.dnn.readNetFromDarknet("D:/YOLO/pretrainedModels/yolov3.cfg","D:/YOLO/pretrainedModels/yolov3.weights")

#detection icin modelimdeki layerlari cekmem gerekiyor. Bunun icin layers icine degiskenleri saklayalim.
layers=model.getLayerNames()
#suan layersta birsuru katman var.
#istedigim sey tum katmanlar degil,detection yapilan katmanlar yani cikti katmanlari.
#yeni yaratacagimiz list'de elemanlar 0'dan basladigi icin verdigi ciktinin 1 eksigi olmasi gerekiyor.
outputLayers=[layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

#simdi modele input olarak BLOB'u verebilirim. 
model.setInput(imgblob)

#daha sonra detection icinde saklanacak degiskeni olusturuyorum.
detectionLayers=model.forward(outputLayers)

#################################### NON MAX SUPPRESSION - PART1 ###########################################

#Simdi non-maximum suppression yontemiyle confidence'i en yuksek kareleri alicam.
# Bu basit bir islem. Hesapladidigim tum BoundingBoxlari ve confidence'lari tutucam.
#Sonra bunlarin icinde tektek gezip. Max olanlari cizdiricem.
#3 tane array olusturucam. 1.si predictedID,2.si boundingBox,3.su de confidence'i tutucak.
IDlist=[]
boxeslist=[]
confidencelist=[]

#################################### END OF PART-1 #########################################################

#%% FOURTH PART
#detection layers icindeki arraylari tektek gezicem ve bu arraylarin icini de tektek gezicem.
for detectionLayer in detectionLayers:
    for objectDetection in detectionLayer:
        scores=objectDetection[5:] #ilk 5 deger boundingBox'la ilgili oldugu icin 5ten sonrasina bak.
        predictedID=np.argmax(scores)
        confidence=scores[predictedID] #guven skoru
        
        #eger confidence > belli bir degerden buyukse cizmesine izin verelim.
        if confidence >0.60:
            label=labels[predictedID] #cizdirmeden once hangi labelda oldugunu bulmasi gerek.
            #ilk 5 deger yeterli degil. YOLO algoritmasiyla alakalı. Genisletmem gerek.
            boundingBox=objectDetection[0:4]*np.array([img_width,img_height,img_width,img_height]) 
            #rectangle cizmek icin
            (boxCenterX,boxCenterY,boxWidth,boxHeight)=boundingBox.astype("int") #boundingboxtan gelen degerlerin int haline esit olucak.
            #rectangle baslangic ve bitis degerleri
            startX=int(boxCenterX - (boxWidth/2))
            startY=int(boxCenterY - (boxHeight/2))
            
#################################### NON MAX SUPPRESSION -PART2 ##################################################
            #listelerin icini dolduralim.
            IDlist.append(predictedID)
            confidencelist.append(float(confidence))
            boxeslist.append([startX,startY,int(boxWidth),int(boxHeight)]) #width ve height int olarak kullanilabiliyor.
 
#################################### END OF PART- 2 ###############################################################


#################################### NON MAX SUPPRESSION -PART3 ##################################################

maxIDs=cv2.dnn.NMSBoxes(boxeslist,confidencelist,0.5,0.4) #max conf sahip boxlari array bicimde dondurur.
#3. parametre guven skoru, 4. parametre threshold. Optimal degerler bunlar.
for max_ID in maxIDs:
    maxClassID=max_ID[0]
    box=boxeslist[maxClassID]
    startX=box[0]
    startY=box[1]
    boxWidth=box[2]
    boxHeight=box[3]
    
                    
                    
    predictedID=IDlist[maxClassID]
    label=labels[predictedID]
    confidence=confidencelist[maxClassID]
                
                    


#################################### END OF PART- 3 ###############################################################

    endX=startX + boxWidth
    endY=startY + boxHeight
            
    #box color
    boxColor=colors[predictedID] #herbir nesne icin farkli bir renk cekmis olucam
    #ama renklerin listede tutulmasi gerekiyor.
    boxColor=[int(each) for each in boxColor]
            
    #confidence'i ekranda da görelim.
    label="{}: {:.2f}%".format(label, confidence*100)
    print("predicted object {}".format(label))
            
            
    cv2.rectangle(img,(startX,startY),(endX,endY),boxColor,1)
    cv2.putText(img,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,boxColor,1)
    #0.5 boyut girdik.
            
cv2.imshow("Detection Window",img)




            
cv2.waitKey(0)
cv2.destroyAllWindows()