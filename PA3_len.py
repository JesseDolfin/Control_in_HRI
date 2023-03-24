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
window = pygame.display.set_mode((1200, 600))   ##twice 600x400 for haptic and VR
pygame.display.set_caption('Virtual Haptic Device')

screenHaptics = pygame.Surface((600,600))
screenVR = pygame.Surface((600,600))

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
cSkin      = (210,161,140)
cFat       = (255,174,66)
cLig_one   = (232,229,221)
cLig_two   = (146,146,146)
cLig_three = (252,228,194)
cFluid     = (255,132,115)
cSpinal    = (255,215,0)
cVerte     = (226,195,152)

cOrange = (255,100,0)
cWhite  = (255,255,255)

# CEREBROSPINAL FLUID
wall_size_factor7 = 0.15     # SPINAL CORD
wall_size_factor8 = 0.0      # VERTEBRAE


####Pseudo-haptics dynamic parameters, k/b needs to be <1
k = .5      ##Stiffness between cursor and haptic display
b = .8       ##Viscous of the pseudohaptic display


##################### Define sprites #####################

# Load in transparant object images and convert to alpha channel
vertebrae_layer  = pygame.image.load("vertebra_test.png").convert_alpha()
vertebrae_layer  = pygame.transform.scale(vertebrae_layer,(140,140))
needle           = pygame.image.load("surgical needle small.png").convert_alpha()
needle           = pygame.transform.scale(needle,(400,40))
needle_undeformed = needle.copy()

# Create pixel masks for every object 
vertebrae_mask   = pygame.mask.from_surface(vertebrae_layer)
needle_mask      = pygame.mask.from_surface(needle)

# Get the rectangles and obstacle locations for rendering and mask offset
skin_position  = [395, 0]


vertebrae_rect  = vertebrae_layer.get_rect()


# Mask offsets
offsets = [[395, -5]]

haptic  = pygame.Rect(*screenHaptics.get_rect().center, 0, 0).inflate(4, 4)
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


# Initialize total simulation space occupied by human subject (start_pos, end_pos, difference)
simulation_space  = [[300, 600, 300], [0, 600, 600]]

# Initialize adjustable scaling factors to play around with tissue size
wall_size_factor1 = 0.05      # SKIN
wall_size_factor2 = 0.125     # FAT
wall_size_factor3 = 0.025     # SUPRASPINAL LIGAMENT
wall_size_factor4 = 0.4       # INTERSPINAL LIGAMENT
wall_size_factor5 = 0.05      # LIGAMENTUM FLAVUM
wall_size_factor6 = 0.0375    # CEREBROSPINAL FLUID
wall_size_factor7 = 0.075     # SPINAL CORD
wall_size_factor8 = 0.15      # VERTEBRAE ONE
wall_size_factor9 = 0.4       # CARTILAGE
wall_size_factor10 = 0.15     # VERTEBRAE TWO

# Vertical wall layers (x, y, width, height)
wall_layer1  = pygame.Rect(simulation_space[0][0],simulation_space[1][0],wall_size_factor1*(simulation_space[0][2]),simulation_space[1][2])
wall_layer2  = pygame.Rect(wall_layer1[0]+wall_layer1[2],simulation_space[1][0],wall_size_factor2*(simulation_space[0][2]),simulation_space[1][2])
wall_layer3  = pygame.Rect(wall_layer2[0]+wall_layer2[2],simulation_space[1][0],wall_size_factor3*(simulation_space[0][2]),simulation_space[1][2])
wall_layer4  = pygame.Rect(wall_layer3[0]+wall_layer3[2],simulation_space[1][0],wall_size_factor4*(simulation_space[0][2]),simulation_space[1][2])
wall_layer5  = pygame.Rect(wall_layer4[0]+wall_layer4[2],simulation_space[1][0],wall_size_factor5*(simulation_space[0][2]),simulation_space[1][2])
wall_layer6  = pygame.Rect(wall_layer5[0]+wall_layer5[2],simulation_space[1][0],wall_size_factor6*(simulation_space[0][2]),simulation_space[1][2])
wall_layer7  = pygame.Rect(wall_layer6[0]+wall_layer6[2],simulation_space[1][0],wall_size_factor7*(simulation_space[0][2]),simulation_space[1][2])
wall_layer8  = pygame.Rect(wall_layer7[0]+wall_layer7[2],simulation_space[1][0],wall_size_factor6*(simulation_space[0][2]),simulation_space[1][2])
wall_layer9  = pygame.Rect(wall_layer8[0]+wall_layer8[2],simulation_space[1][0],wall_size_factor3*(simulation_space[0][2]),simulation_space[1][2])
wall_layer10 = pygame.Rect(wall_layer9[0]+wall_layer9[2],simulation_space[1][0],wall_size_factor9*(simulation_space[0][2]),simulation_space[1][2])
wall_layer11 = pygame.Rect(wall_layer10[0]+wall_layer10[2],simulation_space[1][0],wall_size_factor3*(simulation_space[0][2]),simulation_space[1][2])

# Vertebrae 
wall_layer12 = pygame.Rect(wall_layer9[0]+wall_layer9[2],simulation_space[1][0],wall_size_factor9*(simulation_space[0][2]),wall_size_factor8*simulation_space[1][2])
wall_layer13 = pygame.Rect(wall_layer9[0]+wall_layer9[2],wall_size_factor8*simulation_space[1][2]+30,wall_size_factor9*(simulation_space[0][2]),wall_size_factor8*simulation_space[1][2])
wall_layer14 = pygame.Rect(wall_layer9[0]+wall_layer9[2],wall_layer13[1] + wall_size_factor8*simulation_space[1][2]+30,wall_size_factor9*(simulation_space[0][2]),wall_size_factor8*simulation_space[1][2])
wall_layer15 = pygame.Rect(wall_layer9[0]+wall_layer9[2],wall_layer14[1] + wall_size_factor8*simulation_space[1][2]+30,wall_size_factor9*(simulation_space[0][2]),wall_size_factor8*simulation_space[1][2])
wall_layer16 = pygame.Rect(wall_layer9[0]+wall_layer9[2],wall_layer15[1] + wall_size_factor8*simulation_space[1][2]+30,wall_size_factor9*(simulation_space[0][2]),wall_size_factor8*simulation_space[1][2])

# # Store all objects in dictonary
objects = {'Skin': wall_layer1, 'Fat': wall_layer2, 'Supraspinal ligament one': wall_layer3, 'Interspinal ligament': wall_layer4,
           'Ligamentum flavum': wall_layer5, 'Cerebrospinal fluid one': wall_layer6, 'Spinal cord': wall_layer7,
            'Cerebrospinal fluid two': wall_layer8, 'Supraspinal ligament two':  wall_layer9, 'Cartilage': wall_layer10,
              'Supraspinal ligament three': wall_layer11, 'Vertebrae one': wall_layer12, 'Vertebrae two': wall_layer13,
              'Vertebrae three': wall_layer14, 'Vertebrae four': wall_layer15, 'Vertebrae five': wall_layer16}


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


'''Intial variable declerations'''
K = np.diag([1000,1000]) # stiffness matrix N/m
p = np.array([0.1,0.1]) # actual endpoint position
dp = np.zeros(2) # actual endpoint velocity
ddp = np.zeros(2)
phold = np.zeros(2)
F = np.zeros(2) # endpoint force
m = 0.5
i = 0
t = 0.0 # time

dt = 0.01 # intergration step timedt = 0.01 # integration step time
dts = dt*1 # desired simulation step time (NOTE: it may not be achieved)

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

            ##Rotate the needle
            if event.key ==ord('r'):
                needle_rotation +=1
            if event.key ==ord('e'):
                needle_rotation -= 1

            #visualisation of walls
            if event.key ==ord('o'):
                visiualse_walls = True
            if event.key ==ord('p'):
                visiualse_walls = False


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
        pass

    '''Force calculation'''
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


    # #get wall position of needle
    # tip_needle = pygame.Rect(xh[0]+325,xh[1]+13,2,2)
    
    # # Get needle pos
    # needle_pos = [haptic.topleft[0],haptic.topleft[1]]
    
    # # offset_list = []
    # # for offset in offsets:
    # #     distance = [needle_pos[0]-offset[0], needle_pos[1]-offset[1]]
    # #     offset_list.append(distance)
    
    

    # COMPUTE COLLISIONS AND PRINT WHICH TYPE OF COLLISION TO CONSOLE

    # # Check if overlap occurs betweens masks
    # collision1 = skin_mask.overlap(needle_mask, offset_list[0])

    for value in objects:
        Collision = haptic.colliderect(objects[value])
        if Collision:
            print("Oh no! We touched:", value)

    ######### Send forces to the device #########
    if port:
        F[1] = -F[1]  ##Flips the force on the Y=axis 

        ##Update the forces of the device
        device.set_device_torques(F)
        device.device_write_torques()
        #pause for 1 millisecond
        time.sleep(0.001)
    else:
        ddp = F/m
        dp += ddp*dt
        p += dp*dt
        t += dt
    
    i += 1
    
    ######### Graphical output #########
    ##Render the haptic surface
    screenHaptics.fill(cWhite)
    
    ##Change color based on effort
    colorMaster = (255,\
         255-np.clip(np.linalg.norm(k*(pr-p)/window_scale)*15,0,255),\
         255-np.clip(np.linalg.norm(k*(pr-p)/window_scale)*15,0,255)) #if collide else (255, 255, 255)

    pygame.draw.rect(screenHaptics, colorMaster, (pr[0]-20,pr[1]-20,40,40),border_radius=4)
    
    
    ######### Robot visualization ###################
    # update individual link position
    if robotToggle:
        robot.createPantograph(screenHaptics,pr)
        
    ### Hand visualisation
    #pygame.draw.line(screenHaptics, (0, 0, 0), (haptic.center),(haptic.center+2*k*(xm-xh)))
    
    ##Render the VR surface
    screenVR.fill(cWhite)

    ### Visualize all components of the simulation

    # Draw all the vertical tissue layers
    pygame.draw.rect(screenVR,cSkin,wall_layer1, border_radius = 2)
    pygame.draw.rect(screenVR,cFat,wall_layer2,  border_radius = 2)
    pygame.draw.rect(screenVR,cLig_one,wall_layer3, border_radius = 2)
    pygame.draw.rect(screenVR,cLig_two,wall_layer4,border_radius = 2)
    pygame.draw.rect(screenVR,cLig_three,wall_layer5,border_radius = 2)
    pygame.draw.rect(screenVR,cFluid,wall_layer6,border_radius = 2)
    pygame.draw.rect(screenVR,cSpinal,wall_layer7,border_radius = 2)
    pygame.draw.rect(screenVR,cFluid,wall_layer8,border_radius = 2)
    pygame.draw.rect(screenVR,cLig_one,wall_layer9, border_radius = 2)
    pygame.draw.rect(screenVR,cLig_two,wall_layer10, border_radius = 2)
    pygame.draw.rect(screenVR,cLig_one,wall_layer11, border_radius = 2)
    
    # Draw all the vertebrae
    pygame.draw.rect(screenVR,cVerte,wall_layer12, border_radius = 4)
    pygame.draw.rect(screenVR,cVerte,wall_layer13, border_radius = 4)
    pygame.draw.rect(screenVR,cVerte,wall_layer14, border_radius = 4)
    pygame.draw.rect(screenVR,cVerte,wall_layer15, border_radius = 4)
    pygame.draw.rect(screenVR,cVerte,wall_layer16, border_radius = 4)

    
    # Draw the masks  
    pygame.draw.rect(screenVR, colorHaptic, (p[0],p[1],1,1), border_radius=8) #draw the needle

    #define center of needle
    pivot = [p[0]+175, p[1]+23]
    offset = pygame.math.Vector2(0, 0)

    rotated_image, rect = rotate(needle_undeformed, needle_rotation, pivot, offset)

    #screenVR.blit(spine,(400,0)) #draw the spine
    screenVR.blit(rotated_image,rect) #draw the needle
    
    vert_pos_one   = [wall_layer3[0],-0.75*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
    vert_pos_two   = [wall_layer3[0],0.1*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
    vert_pos_three = [wall_layer3[0],0.95*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
    vert_pos_four  = [wall_layer3[0],1.8*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
    vert_pos_five  = [wall_layer3[0],2.65*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
    
    screenVR.blit(vertebrae_layer,(vert_pos_one[0],vert_pos_one[1])) #draw the needle
    screenVR.blit(vertebrae_layer,(vert_pos_two[0],vert_pos_two[1]))
    screenVR.blit(vertebrae_layer,(vert_pos_three[0],vert_pos_three[1]))
    screenVR.blit(vertebrae_layer,(vert_pos_four[0],vert_pos_four[1]))
    screenVR.blit(vertebrae_layer,(vert_pos_five[0],vert_pos_five[1]))


    ##Fuse it back together
    window.blit(screenHaptics, (0,0))
    window.blit(screenVR, (600,0))

    ##Print status in  overlay
    if debugToggle: 
        
        text = font.render("FPS = " + str(round(clock.get_fps())) + \
                            "  xm = " + str(np.round(10*xm)/10) +\
                            "  xh = " + str(np.round(10*xh)/10) +\
                            "  fe = " + str(np.round(10*F)/10) \
                            , True, (0, 0, 0), (255, 255, 255))
        window.blit(text, textRect)


    pygame.display.flip()   

    ##Slow down the loop to match FPS
    clock.tick(FPS)

pygame.display.quit()
pygame.quit()

