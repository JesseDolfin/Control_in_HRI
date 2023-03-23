# -*- coding: utf-8 -*-

import pygame
import numpy as np
import math
import matplotlib.pyplot as plt
from pantograph import Pantograph
from pyhapi import Board, Device, Mechanisms
from pshape import PShape
import sys, serial, glob
from serial.tools import list_ports
import time


##################### General Pygame Init #####################
##initialize pygame window
pygame.init()
window = pygame.display.set_mode((1200, 400))   ##twice 600x400 for haptic and VR
pygame.display.set_caption('Virtual Haptic Device')

screenHaptics = pygame.Surface((600,400))
screenVR = pygame.Surface((600,400))

##add nice icon from https://www.flaticon.com/authors/vectors-market
icon = pygame.image.load('robot.png')
pygame.display.set_icon(icon)

##add text on top to debugToggle the timing and forces
font = pygame.font.Font('freesansbold.ttf', 18)

pygame.mouse.set_visible(True)     ##Hide cursor by default. 'm' toggles it
 
##set up the on-screen debugToggle
text = font.render('Virtual Haptic Device', True, (0, 0, 0),(255, 255, 255))
textRect = text.get_rect()
textRect.topleft = (10, 10)

xc,yc = screenVR.get_rect().center ##center of the screen

##initialize "real-time" clock
clock = pygame.time.Clock()
FPS = 100   #in Hertz

##define some colors
cWhite = (255,255,255)
cDarkblue = (36,90,190)
cLightblue = (0,176,240)
cRed = (255,0,0)
cOrange = (255,100,0)
cYellow = (255,255,0)


####Pseudo-haptics dynamic parameters, k/b needs to be <1
k = .5      ##Stiffness between cursor and haptic display
b = .8       ##Viscous of the pseudohaptic display


##################### Define sprites #####################

# Load in transparant object images and convert to alpha channel
skin_layer      = pygame.image.load("skin.png").convert_alpha()
fat_layer       = pygame.image.load("fat.png").convert_alpha()
vertebra_one    = pygame.image.load("vertebra_1.png").convert_alpha()
vertebra_two    = pygame.image.load("vertebra_2.png").convert_alpha()
ligament_one    = pygame.image.load("ligament_1.png").convert_alpha()
ligament_two    = pygame.image.load("ligament_2.png").convert_alpha()
ligament_three  = pygame.image.load("ligament_3.png").convert_alpha()
ligament_four   = pygame.image.load("ligament_4.png").convert_alpha()
spinal_cord     = pygame.image.load("spinal_cord.png").convert_alpha()

needle = pygame.image.load("surgical needle small.png").convert_alpha()
needle = pygame.transform.scale(needle,(200,25))

# Create pixel masks for every object 
skin_mask         = pygame.mask.from_surface(skin_layer)
fat_mask          = pygame.mask.from_surface(fat_layer)
vertebra_one_mask = pygame.mask.from_surface(vertebra_one)
vertebra_two_mask = pygame.mask.from_surface(vertebra_two)
spinal_cord_mask  = pygame.mask.from_surface(spinal_cord)

back_mask         = pygame.mask.from_surface(skin_layer)
needle_mask       = pygame.mask.from_surface(needle)

# Get the rectangles and obstacle locations for rendering and mask offset
skin_position = [400, -10]
fat_position  = [416, -10]




back_rect    = skin_layer.get_rect()
needle_rect  = needle.get_rect()
ox = 400
oy = -10


# wall_layer_one   = pygame.Rect(350,0,20,400)  ## Skin
# wall_layer_two   = pygame.Rect(370,0,50,400)  ## Subcutaneous fat
# wall_layer_three = pygame.Rect(420,0,100,400)  ## Supraspinous ligament
# wall_layer_four  = pygame.Rect(520,0,30,400)  ## Interspinous ligament
# wall_layer_five  = pygame.Rect(550,0,25,400)  ## Ligamentum flavum
# wall_layer_six   = pygame.Rect(575,0,25,400)  ## Dural sheath





haptic  = pygame.Rect(*screenHaptics.get_rect().center, 0, 0).inflate(48, 48)
cursor  = pygame.Rect(0, 0, 5, 5)
colorHaptic = cOrange ##color of the wall

xh = np.array(haptic.center)

##Set the old value to 0 to avoid jumps at init
xhold = 0
xmold = 0

##################### Init Virtual env. #####################
visiualse_walls = True
needle_rotation = 0
collision = False

##################### Detect and Connect Physical device #####################
# USB serial microcontroller program id data:
def serial_ports():
    """ Lists serial port names """
    ports = list(serial.tools.list_ports.comports())

    result = []
    for p in ports:
        try:
            port = p.device
            s = serial.Serial(port)
            s.close()
            if p.description[0:12] == "Arduino Zero":
                result.append(port)
                print(p.description[0:12])
        except (OSError, serial.SerialException):
            pass
    return result

CW = 0
CCW = 1

haplyBoard = Board
device = Device
SimpleActuatorMech = Mechanisms
pantograph = Pantograph
robot = PShape

#########Open the connection with the arduino board#########
port = serial_ports()   ##port contains the communication port or False if no device
if port:
    print("Board found on port %s"%port[0])
    haplyBoard = Board("test", port[0], 0)
    device = Device(5, haplyBoard)
    pantograph = Pantograph()
    device.set_mechanism(pantograph)
    
    device.add_actuator(1, CCW, 2)
    device.add_actuator(2, CW, 1)
    
    device.add_encoder(1, CCW, 241, 10752, 2)
    device.add_encoder(2, CW, -61, 10752, 1)
    
    device.device_set_parameters()
else:
    print("No compatible device found. Running virtual environnement...")
    #sys.exit(1)
    
# conversion from meters to pixels
window_scale = 3

##################### Main Loop #####################
##Run the main loop
##TODO - Perhaps it needs to be changed by a timer for real-time see: 
##https://www.pygame.org/wiki/ConstantGameSpeed

run = True
ongoingCollision = False
fieldToggle = True
robotToggle = True
debugToggle = False

center = np.array([xc,yc])    


# Add walls for collision detection and tissue definition
# wall_layer_one   = pygame.Rect(350,0,20,400)  ## Skin
# wall_layer_two   = pygame.Rect(370,0,50,400)  ## Subcutaneous fat
# wall_layer_three = pygame.Rect(420,0,100,400)  ## Supraspinous ligament
# wall_layer_four  = pygame.Rect(520,0,30,400)  ## Interspinous ligament
# wall_layer_five  = pygame.Rect(550,0,25,400)  ## Ligamentum flavum
# wall_layer_six   = pygame.Rect(575,0,25,400)  ## Dural sheath


# walls = {"skin": [wall_layer_one,wall_layer_two,wall_layer_three,wall_layer_four,wall_layer_five,wall_layer_six],"bone": [wall_layer_one]}


while run:
    #########Process events  (Mouse, Keyboard etc...)#########
    for event in pygame.event.get():
        ##If the window is close then quit 
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYUP:
            if event.key == ord('m'):   ##Change the visibility of the mouse
                pygame.mouse.set_visible(not pygame.mouse.get_visible())  
            if event.key == ord('q'):   ##Force to quit
                run = False            
            '''*********** Student can add more ***********'''
            ##Rotate the needle and the hand of the haptic
        

            if event.key ==ord('o'):
                visiualse_walls = True
            if event.key ==ord('p'):
                visiualse_walls = False

            '''*********** !Student can add more ***********'''

    ######### Read position (Haply and/or Mouse)  #########
    ##Get endpoint position xh
    if port and haplyBoard.data_available():    ##If Haply is present
        #Waiting for the device to be available
        #########Read the motorangles from the board#########
        device.device_read_data()
        motorAngle = device.get_device_angles()
        
        #########Convert it into position#########
        device_position = device.get_device_position(motorAngle)
        xh = np.array(device_position)*1e3*window_scale
        xh[0] = np.round(-xh[0]+300)
        xh[1] = np.round(xh[1]-60)
        xm = xh     ##Mouse position is not used
         
    else:
        ##Compute distances and forces between blocks
        xh = np.clip(np.array(haptic.center),0,599)
        xh = np.round(xh)
        
        ##Get mouse position
        cursor.center = pygame.mouse.get_pos()
        xm = np.clip(np.array(cursor.center),0,599)
    
    '''*********** Student should fill in ***********'''
    # add dynamics of the environment
    fe = np.zeros(2)  ##Environment force is set to 0 initially.

    #get wall position of needle
    tip_needle = pygame.Rect(xh[0]+325,xh[1]+13,2,2)
    
    # Get needle pos
    needle_pos = [haptic.topleft[0],haptic.topleft[1]]
    
    # # Compute offset between two masks
    offset = [needle_pos[0]-ox, needle_pos[1]-oy]

    # Check if overlap occurs betweens masks
    collision = back_mask.overlap(needle_mask, offset)
    if collision:
        print("collision")
    if not collision:
        print("no collision")
    # if pygame.sprite.collide_mask(back_mask, needle_mask):
    #     print("collide")

    '''*********** !Student should fill in ***********'''
    
    ##Update old samples for velocity computation
    xhold = xh
    xmold = xm
    
    ######### Send forces to the device #########
    if port:
        fe[1] = -fe[1]  ##Flips the force on the Y=axis 

        ##Update the forces of the device
        device.set_device_torques(fe)
        device.device_write_torques()
        #pause for 1 millisecond
        time.sleep(0.001)
    else:
        ######### Update the positions according to the forces ########
        ##Compute simulation (here there is no inertia)
        ##If the haply is connected xm=xh and dxh = 0
        dxh = (k/b*(xm-xh)/window_scale -fe/b)    ####replace with the valid expression that takes all the forces into account
        dxh = dxh*window_scale
        xh = np.round(xh+dxh)             ##update new positon of the end effector
        
    haptic.center = xh 
    
    ######### Graphical output #########
    ##Render the haptic surface
    screenHaptics.fill(cWhite)
    
    ##Change color based on effort
    colorMaster = (255,\
         255-np.clip(np.linalg.norm(k*(xm-xh)/window_scale)*15,0,255),\
         255-np.clip(np.linalg.norm(k*(xm-xh)/window_scale)*15,0,255)) #if collide else (255, 255, 255)

    pygame.draw.rect(screenHaptics, colorMaster, haptic,border_radius=4)
    
    
    ######### Robot visualization ###################
    # update individual link position
    if robotToggle:
        robot.createPantograph(screenHaptics,xh)
        
    ### Hand visualisation
    pygame.draw.line(screenHaptics, (0, 0, 0), (haptic.center),(haptic.center+2*k*(xm-xh)))
    
    ##Render the VR surface
    screenVR.fill(cWhite)
    '''*********** Student should fill in ***********'''
    ### here goes the visualisation of the VR sceen. 

    


    


    screenVR.blit(skin_layer,(395,-5)) #draw the spine
    screenVR.blit(fat_layer,(402,-5)) #draw the spine
    screenVR.blit(ligament_one,(441,-5)) #draw the spine


    screenVR.blit(ligament_two,(462,-7)) #draw the spine
    screenVR.blit(vertebra_two,(455,-5)) #draw the spine
    # screenVR.blit(needle,(haptic.topleft[0],haptic.topleft[1])) #draw the needle

    screenVR.blit(ligament_three,(545,-5))  #draw the first layer
    screenVR.blit(ligament_four,(525,-6))  #draw the first layer

    screenVR.blit(spinal_cord,(552,-2)) #draw the spine

    # if visiualse_walls == True:
    #     #skin
    #     pygame.draw.rect(screenVR,cRed,wall_layer_one)
    #     pygame.draw.rect(screenVR,cDarkblue,wall_layer_two)
    #     pygame.draw.rect(screenVR,cRed,wall_layer_three)
    #     pygame.draw.rect(screenVR,cDarkblue,wall_layer_four)

        # #bone
        # pygame.draw.rect(screenVR,cDarkblue,wall_layer_one)

        # #draw tip of needle
        # pygame.draw.rect(screenVR,cRed,tip_needle)

        
    
    
    '''*********** !Student should fill in ***********'''

    ##Fuse it back together
    window.blit(screenHaptics, (0,0))
    window.blit(screenVR, (600,0))

    ##Print status in  overlay
    if debugToggle: 
        
        text = font.render("FPS = " + str(round(clock.get_fps())) + \
                            "  xm = " + str(np.round(10*xm)/10) +\
                            "  xh = " + str(np.round(10*xh)/10) +\
                            "  fe = " + str(np.round(10*fe)/10) \
                            , True, (0, 0, 0), (255, 255, 255))
        window.blit(text, textRect)


    pygame.display.flip()    
    ##Slow down the loop to match FPS
    clock.tick(FPS)

pygame.display.quit()
pygame.quit()

