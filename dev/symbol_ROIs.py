import matplotlib.pyplot as plt
import cv2

def main():
    source_image = cv2.imread('screenshots/dragon/1.png')
    print("Select the area")
    area = (cv2.selectROI("Select the area",source_image))
    x = area[0]
    y = area[1]
    width = area[2]
    height = area[3]
    row = int(input("Input the row amount:"))
    col = int(input("Input the column amount:"))
    
    width-=width%col
    height-=height%row
    symbol_width = width/col
    symbol_height = height/row

    symbol_width = width/col
    symbol_height = height/row
    
    frames = []
    for i in range(col):
        for j in range(row):
            frames.append([int(x+symbol_width*i),int(y+symbol_height*j),int(symbol_width),int(symbol_height)])
    
    output = open("game_symbol_ROIs.json","w")
    print(frames,file=output)
    output.close()

    # Debug

    for i in range(len(frames)):
        bbox=frames[i]
        if(bbox):
            x1=bbox[0]
            y1=bbox[1]
            x2=bbox[0]+bbox[2]
            y2=bbox[1]+bbox[3]
            cv2.rectangle(source_image,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(source_image, "({0},{1})".format(int(i%row),int(i/row)), (x1+5,y1-10), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.rectangle(source_image,(x,y),(x+width,y+height),(255,0,0),3)
    # cv2.putText(source_image, "Selected area", (x-10,y-40), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0), 2, cv2.LINE_AA)
  
    print(len(frames),frames)
    plt.figure(figsize=(20,20))
    plt.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show() 

main()