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


def rotate(surface, angle, pivot, offset):
    """Rotate the surface around the pivot point.

    Args:
        surface (pygame.Surface): The surface that is to be rotated.
        angle (float): Rotate by this angle.
        pivot (tuple, list, pygame.math.Vector2): The pivot point.
        offset (pygame.math.Vector2): This vector is added to the pivot.
    """
    rotated_image = pygame.transform.rotozoom(surface, -angle, 1)  # Rotate the image.
    rotated_offset = offset.rotate(angle)  # Rotate the offset vector.
    # Add the offset vector to the center/pivot point to shift the rect.
    rect = rotated_image.get_rect(center=pivot+rotated_offset)
    return rotated_image, rect  # Return the rotated image and shifted rect



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


##################### Define sprites #####################
##define sprites
hhandle = pygame.image.load('handle.png')
hhandle_undeformed = hhandle.copy()
needle = pygame.image.load('surgical needle small.png')
needle = pygame.transform.scale(needle,(350,46))
needle_undeformed = needle.copy()
spine = pygame.image.load('lumbar_spine.png')
spine = pygame.transform.scale(spine,(200,400))
haptic  = pygame.Rect(*screenHaptics.get_rect().center, 0, 0).inflate(48, 48)
cursor  = pygame.Rect(0, 0, 5, 5)
colorHaptic = cOrange ##color of the wall

xh = np.array(haptic.center)

##Set the old value to 0 to avoid jumps at init
xhold = 0
xmold = 0

##################### Init Virtual env. #####################
visiualse_walls = False
needle_rotation = 0
collision = False


k = 0.5


CW = 0
CCW = 1

haplyBoard = Board
device = Device
SimpleActuatorMech = Mechanisms
pantograph = Pantograph
robot = PShape

# conversion from meters to pixels
window_scale = 3

##################### Main Loop #####################
##Run the main loop
##TODO - Perhaps it needs to be changed by a timer for real-time see: 
##https://www.pygame.org/wiki/ConstantGameSpeed

#add booleans
run = True
ongoingCollision = False
fieldToggle = True
robotToggle = True
debugToggle = False

# initial conditions
K = np.diag([1000,1000]) # stiffness matrix N/m
p = np.array([0.1,0.1]) # actual endpoint position
dp = np.zeros(2) # actual endpoint velocity
F = np.zeros(2) # endpoint force
m = 0.5
i = 0
t = 0.0 # time

center = np.array([xc,yc])    


#add walls for collision detection
wall_skin_1  = pygame.Rect(430,0,4,110)
wall_skin_2  = pygame.Rect(440,110,4,155)
wall_skin_3  = pygame.Rect(430,265,4,85)
wall_skin_4  = pygame.Rect(420,350,4,50)
wall_bone_1 = pygame.Rect(467,180,43,45)

walls = {"skin": [wall_skin_1,wall_skin_2,wall_skin_3,wall_skin_4],"bone": [wall_bone_1]}


# SIMULATION PARAMETERS
dt = 0.01 # intergration step timedt = 0.01 # integration step time
dts = dt*1 # desired simulation step time (NOTE: it may not be achieved)

while run:
    '''#########Process events  (Mouse, Keyboard etc...)#########'''
    for event in pygame.event.get():
        ##If the window is close then quit 
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYUP:
            if event.key == ord('m'):   ##Change the visibility of the mouse
                pygame.mouse.set_visible(not pygame.mouse.get_visible())  
            if event.key == ord('q'):   ##Force to quit
                run = False            
          
            ##Rotate the needle 
            if event.key == ord('r'):
                needle_rotation += 1
              
            if event.key == ord('e'):
                needle_rotation -= 1

            if event.key ==ord('o'):
                visiualse_walls = True
            if event.key ==ord('p'):
                visiualse_walls = False

    '''Dynamics'''
    # add dynamics of the environment
    F = np.zeros(2)  ##Environment force is set to 0 initially.

    cursor = pygame.mouse.get_pos() 
    pr = cursor # in the first case cursor pos is also reference pos (will change if we do this via udp)

    # spring force calculation
    Fs = K @ (pr-p) 
    
    # damping force calculation
    D = 2*0.7*np.sqrt(K)
    Fd = D @ dp
   
    #endpoint force
    F = Fs - Fd
    
    ddp = F/m
    dp += ddp*dt
    p += dp*dt
    t += dt

    i = i + 1
    

    ######### Graphical output #########
    ##Render the haptic surface
    screenHaptics.fill(cWhite)
    
    ##Change color based on effort
    colorMaster = (255,\
         255-np.clip(np.linalg.norm(k*(p-pr)/window_scale)*15,0,255),\
         255-np.clip(np.linalg.norm(k*(p-pr)/window_scale)*15,0,255)) #if collide else (255, 255, 255)

    pygame.draw.rect(screenHaptics, colorMaster, (pr[0]-20,pr[1]-20,40,40),border_radius=4)
 


    ######### Robot visualization ###################
    # update individual link position
    if robotToggle:
        robot.createPantograph(screenHaptics,pr)
        


    '''here goes the visualisation of the VR sceen'''
    ##Render the VR surface
    screenVR.fill(cWhite)
    #define center of needle
    pivot = [p[0]+175, p[1]+23]
    offset = pygame.math.Vector2(0, 0)

    rotated_image, rect = rotate(needle_undeformed, needle_rotation, pivot, offset)

    screenVR.blit(spine,(400,0)) #draw the spine
    screenVR.blit(rotated_image,rect) #draw the needle

    #visualisation of collision boxes
    if visiualse_walls == True:
        #skin
        pygame.draw.rect(screenVR,cRed,wall_skin_1)
        pygame.draw.rect(screenVR,cRed,wall_skin_2)
        pygame.draw.rect(screenVR,cRed,wall_skin_3)
        pygame.draw.rect(screenVR,cRed,wall_skin_4)

        #bone
        pygame.draw.rect(screenVR,cDarkblue,wall_bone_1)
    
    
    pygame.draw.rect(screenVR,cRed,(p[0],p[1],1,1))
    ##Fuse it back together
    window.blit(screenHaptics, (0,0))
    window.blit(screenVR, (600,0))

    
    pygame.display.flip()    
    ##Slow down the loop to match FPS
    clock.tick(FPS)

pygame.display.quit()
pygame.quit()

