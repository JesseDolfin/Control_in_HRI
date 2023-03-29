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
import pandas as pd

##################### General Pygame Init #####################
def rotMat(angle):
    transformation_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),  np.cos(angle)]])
    return transformation_matrix


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
center = np.array([xc,yc]) 

##initialize "real-time" clock
clock = pygame.time.Clock()
FPS = 100   #in Hertz

## Define colors to be used to render different tissue layers and haptic
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

####Pseudo-haptics dynamic parameters, k/b needs to be <1
K_TISSUE = .5      ##Stiffness between cursor and haptic display
D = 1.5      ##Viscous of the pseudohaptic display

##################### Define sprites #####################

# Load in transparant object images and convert to alpha channel
vertebrae_layer  = pygame.image.load("vertebra_test.png").convert_alpha()
vertebrae_layer  = pygame.transform.scale(vertebrae_layer,(140,140))

# Create pixel masks for every object 
vertebrae_mask   = pygame.mask.from_surface(vertebrae_layer)

# Get the rectangles and obstacle locations for rendering and mask offset
vertebrae_rect   = vertebrae_mask.get_rect()

haptic  = pygame.Rect(*screenHaptics.get_rect().center, 0, 0).inflate(40,40)
cursor  = pygame.Rect(0, 0, 5, 5)
colorHaptic = cOrange ##color of the wall

'''Init all variables'''
xh = np.array(haptic.center,dtype='int32')
dxh = np.zeros(2)
xhold = np.zeros(2)
phold = np.zeros(2)

damping = np.zeros(2)
K = np.diag([1000,1000]) # stiffness matrix N/m
dt = 0.01 # intergration step timedt = 0.01 # integration step time
i = 0 # loop counter
t = 0 # time

# Metrics
max_force_exerted = np.zeros(2)
bone_collision_count = 0
record_deviation_y = []
xhhold = np.zeros(2)
spinal_coord_collision_hit = False

# Init Virtual env.
needle_rotation = 0
alpha = 0

# Declare some simulation booleans to switch between states
robotToggle = True
debugToggle = True
away_from_bone = True
   
# Set all environment parameters to simulate damping in the various tissue layers
D_TISSUE_SKIN   = 5
D_TISSUE_FAT    = 10
D_TISSUE_SUPRA  = 5
D_TISSUE_INTER  = 5
D_TISSUE_FLAVUM = 5
D_TISSUE_FLUID  = 10
D_TISSUE_CORD   = 5
D_TISSUE_CART   = 5

#  Set all environment parameters to simulate damping in the various tissue layers
MAX_TISSUE_SKIN     = 1
MAX_TISSUE_FAT      = 2 
MAX_TISSUE_SUPRA    = 3
MAX_TISSUE_INTER    = 2
MAX_TISSUE_FLAVUM   = 1
MAX_TISSUE_FLUID    = 0.5
MAX_TISSUE_CORD     = 1
MAX_TISSUE_CART     = 5

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

# Vertebrae layers modelled as rectangles
wall_layer12 = pygame.Rect(wall_layer9[0]+wall_layer9[2],simulation_space[1][0],wall_size_factor9*(simulation_space[0][2]),wall_size_factor8*simulation_space[1][2])
wall_layer13 = pygame.Rect(wall_layer9[0]+wall_layer9[2],wall_size_factor8*simulation_space[1][2]+30,wall_size_factor9*(simulation_space[0][2]),wall_size_factor8*simulation_space[1][2])
wall_layer14 = pygame.Rect(wall_layer9[0]+wall_layer9[2],wall_layer13[1] + wall_size_factor8*simulation_space[1][2]+30,wall_size_factor9*(simulation_space[0][2]),wall_size_factor8*simulation_space[1][2])
wall_layer15 = pygame.Rect(wall_layer9[0]+wall_layer9[2],wall_layer14[1] + wall_size_factor8*simulation_space[1][2]+30,wall_size_factor9*(simulation_space[0][2]),wall_size_factor8*simulation_space[1][2])
wall_layer16 = pygame.Rect(wall_layer9[0]+wall_layer9[2],wall_layer15[1] + wall_size_factor8*simulation_space[1][2]+30,wall_size_factor9*(simulation_space[0][2]),wall_size_factor8*simulation_space[1][2])

# Store all objects in a dict which can be accessed for collision detection
objects_dict = {'Skin': wall_layer1, 'Fat': wall_layer2, 'Supraspinal ligament one': wall_layer3, 'Interspinal ligament': wall_layer4,
                'Ligamentum flavum': wall_layer5, 'Cerebrospinal fluid one': wall_layer6, 'Spinal cord': wall_layer7,
                'Cerebrospinal fluid two': wall_layer8, 'Supraspinal ligament two':  wall_layer9, 'Cartilage': wall_layer10,
                'Supraspinal ligament three': wall_layer11, 'Vertebrae one': wall_layer12, 'Vertebrae two': wall_layer13,
                'Vertebrae three': wall_layer14, 'Vertebrae four': wall_layer15, 'Vertebrae five': wall_layer16}

# Initialize a collision dictionary to store booleans corresponding to all objects that are in collision
collision_dict = {'Skin': False, 'Fat': False, 'Supraspinal ligament one': False, 'Interspinal ligament': False,
                    'Ligamentum flavum': False, 'Cerebrospinal fluid one': False, 'Spinal cord': False,
                    'Cerebrospinal fluid two': False, 'Supraspinal ligament two':  False, 'Cartilage': False,
                    'Supraspinal ligament three': False, 'Vertebrae one': False, 'Vertebrae two': False,
                    'Vertebrae three': False, 'Vertebrae four': False, 'Vertebrae five': False}

# Initialize a dictonary which holds all the simulation parameters for efficiency
variable_dict = {'Skin': {'D_TISSUE': D_TISSUE_SKIN, 'max_tissue_force': MAX_TISSUE_SKIN, 'collision_bool': True, 'update_bool': True, 'penetration_bool': False},
                    'Fat' : {'D_TISSUE': D_TISSUE_FAT,'max_tissue_force': MAX_TISSUE_FAT, 'collision_bool': True, 'update_bool': True,'penetration_bool': False},
                    'Supraspinal ligament one': {'D_TISSUE': D_TISSUE_SUPRA,'max_tissue_force': MAX_TISSUE_SUPRA, 'collision_bool': True, 'update_bool': True,'penetration_bool': False},
                    'Interspinal ligament': {'D_TISSUE': D_TISSUE_INTER,'max_tissue_force': MAX_TISSUE_INTER, 'collision_bool': True, 'update_bool': True,'penetration_bool': False},
                    'Ligamentum flavum': {'D_TISSUE': D_TISSUE_FLAVUM,'max_tissue_force': MAX_TISSUE_FLAVUM, 'collision_bool': True, 'update_bool': True,'penetration_bool': False},
                    'Cerebrospinal fluid one': {'D_TISSUE': D_TISSUE_FLUID,'max_tissue_force': MAX_TISSUE_FLUID, 'collision_bool': True, 'update_bool': True,'penetration_bool': False},
                    'Spinal cord': {'D_TISSUE': D_TISSUE_CORD,'max_tissue_force': MAX_TISSUE_CORD, 'collision_bool': True, 'update_bool': True,'penetration_bool': False},
                    'Cerebrospinal fluid two': {'D_TISSUE': D_TISSUE_FLUID,'max_tissue_force': MAX_TISSUE_FLUID , 'collision_bool': True, 'update_bool': True,'penetration_bool': False},
                    'Supraspinal ligament two': {'D_TISSUE': D_TISSUE_SUPRA,'max_tissue_force': MAX_TISSUE_SUPRA, 'collision_bool': True, 'update_bool': True,'penetration_bool': False},
                    'Cartilage': {'D_TISSUE': D_TISSUE_CART,'max_tissue_force': MAX_TISSUE_CART, 'collision_bool': True, 'update_bool': True,'penetration_bool': False},
                    'Supraspinal ligament three': {'D_TISSUE': D_TISSUE_SUPRA,'max_tissue_force': MAX_TISSUE_SUPRA, 'collision_bool': True, 'update_bool': True,'penetration_bool': False}}


# Compose the rectangles belonging to every vertebrae
vert_rect1 = [wall_layer3[0],-0.75*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
vert_rect2 = [wall_layer3[0],0.1*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
vert_rect3 = [wall_layer3[0],0.95*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
vert_rect4 = [wall_layer3[0],1.8*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
vert_rect5 = [wall_layer3[0],2.65*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]

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

# haply stuff
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
while run:
    penetration = True
    collision_bone = False
    collision_any = False
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
                needle_rotation +=5
                alpha += np.deg2rad(5)
            if event.key ==ord('e'):
                needle_rotation -= 5
                alpha -= np.deg2rad(5)
                


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
        cursor = pygame.mouse.get_pos()
        xm = np.clip(np.array(cursor),0,599)
    

    ########### COMPUTE COLLISIONS AND PRINT WHICH TYPE OF COLLISION TO CONSOLE ###########
    # Define haptic center and endpoint of our haptic (needle tip)
    haptic_endpoint = pygame.Rect(haptic.center[0]+np.cos(alpha)*250,haptic.center[1]+np.sin(alpha)*250, 1, 1)
    
    # Create endpoint masks for collision detection between endpoint and drawn vertebrae (odd shape so cannot be rendered using rectangles)
    haptic_endpoint_mask = pygame.mask.Mask((haptic_endpoint.width, haptic_endpoint.height))
    haptic_endpoint_mask.fill()

 
    # Compute offset between haptic endpoint and every vertebrae mask
    xoffset1 = vert_rect1[0] - haptic_endpoint[0]
    yoffset1 = vert_rect1[1] - haptic_endpoint[1]
    xoffset2 = vert_rect2[0] - haptic_endpoint[0]
    yoffset2 = vert_rect2[1] - haptic_endpoint[1]
    xoffset3 = vert_rect3[0] - haptic_endpoint[0]
    yoffset3 = vert_rect3[1] - haptic_endpoint[1]
    xoffset4 = vert_rect4[0] - haptic_endpoint[0]
    yoffset4 = vert_rect4[1] - haptic_endpoint[1]
    xoffset5 = vert_rect5[0] - haptic_endpoint[0]
    yoffset5 = vert_rect5[1] - haptic_endpoint[1]

    # Check collision for every vertebrae and endpoint
    vert1_collision = haptic_endpoint_mask.overlap(vertebrae_mask, (xoffset1, yoffset1))
    vert2_collision = haptic_endpoint_mask.overlap(vertebrae_mask, (xoffset2, yoffset2))
    vert3_collision = haptic_endpoint_mask.overlap(vertebrae_mask, (xoffset3, yoffset3))
    vert4_collision = haptic_endpoint_mask.overlap(vertebrae_mask, (xoffset4, yoffset4))
    vert5_collision = haptic_endpoint_mask.overlap(vertebrae_mask, (xoffset5, yoffset5))
    
    # Check if any of the drawn vertebrae are in collision with the needle tip, if so flip boolean to limit needle movement later in pipeline
    if vert1_collision or vert2_collision or vert3_collision or vert4_collision or vert5_collision:
        collision_bone = True

    ########## COMPUTE ENDPOINT FORCE FEEDBACK ##########

    # Initialize zero endpoint force and reference position based on cursor or Haply
    fe = np.zeros(2)
    reference_pos  = cursor
       
    # Update the collision dict to all False to reset collision dict 
    collision_dict.update((value, False) for value in collision_dict)
    collision_flag = False
    
    # Loop over all the objects and check for collision
    for value in objects_dict:
        Collision = haptic_endpoint.colliderect(objects_dict[value])

        # If collision is detected in specific layer only update this value to True
        if Collision:
            collision_any = True
            collision_dict.update({value: True})

    # Check which values in collision dict are True and update endpoint force and damping factor accordingly to simulate different tissues
    # Based on SOURCE we will implement a multilayer force displacement model as follows:
    # ~ Pre-puncture:  R_F = S_f 
    # ~ Post-puncture: R_F = C_f + F_f
      
    # Compute endpoint velocity and update previous haptic state
    endpoint_velocity = (xhold - xh)/FPS
    xhold = xh

    # Loop over all the rectangular objects and check for collision, note that we exclude the vertebrae from the loop
    # The reason being that these are modelled infinitely stiff so they don't need damping etc.
    Bones = {'Vertebrae one', 'Vertebrae two', 'Vertebrae three', 'Vertebrae four', 'Vertebrae five'}
    for collision in collision_dict:
        if collision not in Bones and collision_dict[collision] == True:
            
            # For the objects(tissues) in collision with the needle tip set the damping value of the environment accordingly
            # Additionally, flip positional and collision boolean
            damping        = K * variable_dict[collision]['D_TISSUE']
            collision_bool = variable_dict[collision]['collision_bool']
            update_bool    = variable_dict[collision]['update_bool']
    
    # In case no collisions are detected default the damping value of the environment to zero
    if all(value == False for value in collision_dict.values()):
        damping = np.zeros(2)
    
    # Check if any of the rectangular vertebrae are in collision, if so flip bone collision boolean to limit needle movement
    if collision_dict['Vertebrae one'] or collision_dict['Vertebrae two'] or collision_dict['Vertebrae three'] or collision_dict['Vertebrae four'] or collision_dict['Vertebrae five']:
        collision_bone = True
    else:
        pass
    
    # Compute the endpoint force which acts at needle tip
    fe = K @ (xm-xh) - (2*0.7*np.sqrt(K) @ dxh)

    #find maximum force exerted
    if i>120:
        if fe[0] > max_force_exerted[0]:
            max_force_exerted[0] = fe[0]
        if fe[1] > max_force_exerted[1]:
            max_force_exerted[1] = fe[1]

    #print("xm:",xm,"xh:",xh,"fe:",fe,"dxh:",dxh)
    # Compute damping force
    fd = -damping @ endpoint_velocity

 
    ######### Send computed forces to the device #########
    if port:
        fe[1] = -fe[1]  ##Flips the force on the Y=axis 

        ##Update the forces of the device
        device.set_device_torques(fe)
        device.device_write_torques()
        #pause for 1 millisecond
        time.sleep(0.001)
   
    else: 
        ddxh = fe 
        
        #update velocity to accomodate damping
        dxh += ddxh*dt - fd

        # In case collision occurs with vertebrae simulate an infinitely stiff bone
        if collision_bone and away_from_bone:
            phold = xh
            away_from_bone = False
            bone_collision_count += 0.5

        if reference_pos[0] >= phold[0] and not away_from_bone:
            dxh = np.zeros(2)
        else: 
            away_from_bone =  True

        # Loop over the detected collisions dictonary (excluding vertebrae), in case collision is detected retrieve tissue parameters from parameter dict
        Bones = {'Vertebrae one', 'Vertebrae two', 'Vertebrae three', 'Vertebrae four', 'Vertebrae five'}
        for collision in collision_dict:
            if collision not in Bones and collision_dict[collision] == True:
                # Set the maximum tissue force, the maximum force exerted by needle pre-puncture
                max_tissue_force = variable_dict[collision]['max_tissue_force']
                #print("Max tissue force: ", max_tissue_force)
                if collision == 'Spinal cord' and i>120:
                    spinal_coord_collision_hit = True
                # Check if collision has occured and fix the current position of the haptic as long as no puncture has occured 
                if update_bool and i>120:
                    phold = xh
                    variable_dict[collision]['update_bool'] = False

                #find standard deviation
                if i>120:
                    dev = np.abs(xh - xhhold)
                    dx = dev[0]
                    dy = dev[1]

                    dx *= math.sin(-alpha)
                    dy *= math.cos(-alpha)
             
                    record_deviation_y.append([dx,dy])

                penetration_bool = variable_dict[collision]['penetration_bool']

                # Compute total endpoint force applied to haptic by the user and check if it exceeds the penetration threshold
                if not penetration_bool:
                    F_pen = (reference_pos[0]-phold[0])*0.1*math.cos(alpha)
                else:
                    F_pen = 0
                
                if F_pen > max_tissue_force:
                    variable_dict[collision]['penetration_bool'] = True
                    penetration_bool = True
            
                if xh[0] > reference_pos[0]:
                    pass
                elif not penetration_bool:
                    dxh = np.zeros(2)

                
                
        if all(value == False for value in collision_dict.values()):
            for collision in collision_dict:
                if collision not in Bones:
                    variable_dict[collision]['penetration_bool'] = False
                    variable_dict[collision]['update_bool'] = True
                  

                    
        #dxh = (K_TISSUE/D_TISSUE*(xm-xh)/window_scale -fe/D_TISSUE)  ####replace with the valid expression that takes all the forces into account
        #dxh = dxh*window_scale
        #xh = np.round(xh+dxh)             ##update new positon of the end effector 
        # 
        #  
        xhhold = xh
        xh = dxh*dt + xh
        i += 1
        t += dt
    haptic.center = xh 
  
    ######### Graphical output #########
    ##Render the haptic surface
    screenHaptics.fill(cWhite)
    
    ##Change color based on effort
    colorMaster = (255,\
         255-np.clip(np.linalg.norm(K_TISSUE*(xm-xh)/window_scale)*15,0,255),\
         255-np.clip(np.linalg.norm(K_TISSUE*(xm-xh)/window_scale)*15,0,255)) #if collide else (255, 255, 255)
    
    pygame.draw.line(screenHaptics, (0, 0, 0), (haptic.center),(haptic.center+2*K_TISSUE*(xm-xh)))
    pygame.draw.rect(screenHaptics, colorMaster, haptic,border_radius=4)
    
    
    ######### Robot visualization ###################
    # update individual link position
    if robotToggle:
        robot.createPantograph(screenHaptics,xh)
        
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
    vert_pos_one   = [wall_layer3[0],-0.75*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
    vert_pos_two   = [wall_layer3[0],0.1*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
    vert_pos_three = [wall_layer3[0],0.95*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
    vert_pos_four  = [wall_layer3[0],1.8*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
    vert_pos_five  = [wall_layer3[0],2.65*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]

    # Draw all the vertebrae
    screenVR.blit(vertebrae_layer,(vert_pos_one[0],vert_pos_one[1])) 
    screenVR.blit(vertebrae_layer,(vert_pos_two[0],vert_pos_two[1]))
    screenVR.blit(vertebrae_layer,(vert_pos_three[0],vert_pos_three[1]))
    screenVR.blit(vertebrae_layer,(vert_pos_four[0],vert_pos_four[1]))
    screenVR.blit(vertebrae_layer,(vert_pos_five[0],vert_pos_five[1]))  

    # Draw needle 
    pygame.draw.line(screenVR, cOrange, (haptic.center[0],haptic.center[1]), (haptic.center[0]+np.cos(alpha)*250, haptic.center[1]+ np.sin(alpha)*250), 2 )
    pygame.draw.line(screenVR, cOrange, (haptic.center[0],haptic.center[1]), (haptic.center[0]+np.sin(-alpha)*25, haptic.center[1]+ np.cos(-alpha)*25), 2 )
    pygame.draw.line(screenVR, cOrange, (haptic.center[0],haptic.center[1]), (haptic.center[0]-np.sin(-alpha)*25, haptic.center[1]- np.cos(-alpha)*25), 2 )

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

record_deviation_y = np.array(record_deviation_y)

try:
   dy = record_deviation_y[:,1]
   std_y = np.std(dy)

except:
    print("No data recored for std_y, using value of 0")
    std_y = 0


#save metrics to the csv file
d = [['Participant'],
     ['Time taken: ','{0:.2f}'.format(t),' s'],
     ['Distance to fluid: ',(wall_layer6[0] - haptic_endpoint[0]),' pixels'],
     ['Number of bone hits: ',int(bone_collision_count-1)],
     ['Spinal coord hit: ',spinal_coord_collision_hit],
     ['Maximum exerted force: ',max_force_exerted/10000],
     ['Deviation inside of tissue: ',std_y],
     ['']]
df = pd.DataFrame(data=d)
df.to_csv('test_csv.csv',mode='a',header=False,index=False)