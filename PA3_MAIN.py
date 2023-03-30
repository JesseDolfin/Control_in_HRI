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
    transformation_matrix = np.array([[np.cos(-angle), np.sin(-angle)],[-np.sin(-angle),  np.cos(-angle)]])
    return transformation_matrix

def compute_line(begin_pos, end_pos):
    x1 = begin_pos[0]
    x2 = end_pos[0]
    y1 = begin_pos[1]
    y2 = end_pos[1]

    a = (y2-y1)/(x2-x1) #flip gradient due to flipped y
    b = (y2-a*x2)
    
    return a, b


##initialize pygame window
pygame.init()
window = pygame.display.set_mode((1200, 300))   ##twice 600x400 for haptic and VR
pygame.display.set_caption('Virtual Haptic Device')

screenHaptics = pygame.Surface((600,300))
screenVR = pygame.Surface((600,300))

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
FPS = 300   #in Hertz

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
vertebrae_layer  = pygame.transform.scale(vertebrae_layer,(63,85))

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
update_prox = True
a = 0
b = 0
# Init Virtual env.
needle_rotation = 0
alpha = 0

# Declare some simulation booleans to switch between states
robotToggle = True
debugToggle = True
away_from_bone = True
spinal_coord_collision = False
toggle_visual = True
   
# Set all environment parameters to simulate damping in the various tissue layers
D_TISSUE_SKIN   = 14.4 # dens = 1.1kg/L
D_TISSUE_FAT    = 11.8 # dens = 0.9kg/L
D_TISSUE_SUPRA  = 15   # dens = 1.142kg/L
D_TISSUE_INTER  = 15   # dens = 1.142kg/L
D_TISSUE_FLAVUM = 15   # dens = 1.142kg/L
D_TISSUE_FLUID  = 13.2 # dens = 1.007kg/L
D_TISSUE_CORD   = 14   # dens = 1.075/L
D_TISSUE_CART   = 14.4 # dens = 1.1kg/L

#  Set all environment parameters to simulate damping in the various tissue layers
MAX_TISSUE_SKIN     = 1 #6.037
MAX_TISSUE_FAT      = 0.36 #2.2
MAX_TISSUE_SUPRA    = 1.49 #9
MAX_TISSUE_INTER    = 1.24 #7.5
MAX_TISSUE_FLAVUM   = 2 #12.1
MAX_TISSUE_FLUID    = 0.4 #2.4
MAX_TISSUE_CORD     = 0.4 #2.4
MAX_TISSUE_CART     = 5 #

# Initialize total simulation space occupied by human subject (start_pos, end_pos, difference)
simulation_space  = [[400, 600, 300], [0, 300, 700]]

# Initialize adjustable scaling factors to play around with tissue size 2 px / mm
wall_size_factor1 = 0.04      # SKIN , 5.6 mm 
wall_size_factor2 = 0.045     # FAT 
wall_size_factor3 = 1/150     # SUPRASPINAL LIGAMENT 0.72mm
wall_size_factor4 = 0.2       # INTERSPINAL LIGAMENT 30 mm
wall_size_factor5 = 0.03      # LIGAMENTUM FLAVUM 4.5 mm
wall_size_factor6 = 1/37.5    # CEREBROSPINAL FLUID 4 mm
wall_size_factor7 = 0.1       # SPINAL CORD 15 mm
wall_size_factor8 = 1/13      # VERTEBRAE ONE 11.5 mm
wall_size_factor9 = 0.393333  # CARTILAGE disk 118 mm
wall_size_factor10 = 1/11.5   # VERTEBRAE TWO 

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
wall_layer17 = pygame.Rect(wall_layer9[0]+wall_layer9[2],wall_layer16[1] + wall_size_factor8*simulation_space[1][2]+30,wall_size_factor9*(simulation_space[0][2]),wall_size_factor8*simulation_space[1][2])

# Store all objects in a dict which can be accessed for collision detection
objects_dict = {'Skin': wall_layer1, 'Fat': wall_layer2, 'Supraspinal ligament one': wall_layer3, 'Interspinal ligament': wall_layer4,
                'Ligamentum flavum': wall_layer5, 'Cerebrospinal fluid one': wall_layer6, 'Spinal cord': wall_layer7,
                'Cerebrospinal fluid two': wall_layer8, 'Supraspinal ligament two':  wall_layer9, 'Cartilage': wall_layer10,
                'Supraspinal ligament three': wall_layer11, 'Vertebrae one': wall_layer12, 'Vertebrae two': wall_layer13,
                'Vertebrae three': wall_layer14, 'Vertebrae four': wall_layer15, 'Vertebrae five': wall_layer16,'Vertebrae six':wall_layer17}

# Initialize a collision dictionary to store booleans corresponding to all objects that are in collision
collision_dict = {'Skin': False, 'Fat': False, 'Supraspinal ligament one': False, 'Interspinal ligament': False,
                    'Ligamentum flavum': False, 'Cerebrospinal fluid one': False, 'Spinal cord': False,
                    'Cerebrospinal fluid two': False, 'Supraspinal ligament two':  False, 'Cartilage': False,
                    'Supraspinal ligament three': False, 'Vertebrae one': False, 'Vertebrae two': False,
                    'Vertebrae three': False, 'Vertebrae four': False, 'Vertebrae five': False,'Vertebrae six':False}

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
vert_rect1 = [wall_layer3[0],-0.7*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
vert_rect2 = [wall_layer3[0],0.3*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
vert_rect3 = [wall_layer3[0],1.3*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
vert_rect4 = [wall_layer3[0],2.3*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
vert_rect5 = [wall_layer3[0],3.3*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]
vert_rect6 = [wall_layer3[0],4.3*vertebrae_rect[3]+wall_size_factor8*simulation_space[1][2]]

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

            # Toggle between visual feedback or not
            if event.key ==ord('v'):
                toggle_visual = not toggle_visual
                


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
        xh = np.array(haptic.center)
        
        ##Get mouse position
        cursor = pygame.mouse.get_pos()
        xm = np.array(cursor) 
    
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
      
    # Compute endpoint velocity and update previous haptic state
    endpoint_velocity = (xhold - xh)/FPS
    xhold = xh
   
    # Loop over all the rectangular objects and check for collision, note that we exclude the vertebrae from the loop
    # The reason being that these are modelled infinitely stiff so they don't need damping etc.
    Bones = {'Vertebrae one', 'Vertebrae two', 'Vertebrae three', 'Vertebrae four', 'Vertebrae five','Vertebrae six'}
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
    if collision_dict['Vertebrae one'] or collision_dict['Vertebrae two'] or collision_dict['Vertebrae three'] or collision_dict['Vertebrae four'] or collision_dict['Vertebrae five'] or collision_dict['Vertebrae six']:
        collision_bone = True
    else:
        pass
    
    # Compute the endpoint force which acts at needle tip
    fe = (K @ (xm-xh) - (2*0.7*np.sqrt(np.abs(K)) @ dxh)) 
    
    # Fix the proxy position of needle contact point to skin and update only when no contact is made with any tissue
    if update_prox and collision_dict['Skin']:
        begin_pos = (haptic.center[0],haptic.center[1])
        end_pos   = (haptic.center[0]+np.cos(alpha)*250, haptic.center[1]+ np.sin(alpha)*250)
        a,b = compute_line(begin_pos, end_pos)

        update_prox = False

    # Enable proxy update as soon as no contact is made with any tissue
    if all(value == False for value in collision_dict.values()):
        update_prox = True

    # Compute the distance to a line along the needle which updates as soon as contact with skin is made.
    if any(value == True for value in collision_dict.values()):
        
        distance_from_line = (a*(xm[0]+np.cos(alpha)*250)-1*(xm[1]+ np.sin(alpha)*250) +b)/np.sqrt(a**2+(-1)**2)

        # Tissue stiffness matrix which gives a feedback force depending on how far reference pos for needle is from needle orientation
        tissue_stiffness_matrix = np.diag([10,1000])

        # Compute the force and scale with respective angle along x and y axis.
        needle_offset_force = (tissue_stiffness_matrix * distance_from_line)*np.array([np.sin(alpha), np.cos(alpha)])

        # Add the needle_offset_force to the endpoint force 
        fe += [needle_offset_force[0,0], needle_offset_force[1,1]]  

        # We will use the needle offset force to approximate the normal force exerted on the needle by the tissue layers.
        # This enables us to implement kinetic friction (note that for every layer passed the kinetic friction increases as the amount of tissues exerting friction increases)
    
        if alpha == 0:

            extra_tissue_stiffness_matrix = np.diag([1000,0])

            kinetic_friction_coefficient = 0.46

            tissue_normal_force_x = np.abs((extra_tissue_stiffness_matrix * distance_from_line))[1,1]
            tissue_normal_force_y = np.abs((extra_tissue_stiffness_matrix * distance_from_line))[0,0]
            
            # Note that the kinetic friction is based on normal force so F_x = mu_kinetic * Fn_y and F_y = mu_kinetic * Fn_x
            frictional_force = tissue_normal_force_x*kinetic_friction_coefficient
            fe[0] += -frictional_force
            
        elif alpha !=0:
            extra_tissue_stiffness_matrix = np.diag([1000,1000])
            kinetic_friction_coefficient = 0.46
            
            tissue_normal_force = (extra_tissue_stiffness_matrix * distance_from_line)*np.array([np.cos(alpha), np.sin(alpha)])
            
        
            # Note that the kinetic friction is based on normal force so F_x = mu_kinetic * Fn_y and F_y = mu_kinetic * Fn_x
            frictional_force = kinetic_friction_coefficient*tissue_normal_force

            fe[0] += frictional_force[0,0]
            fe[1] += frictional_force[1,1]

    # Now that we have the normal force exerted by the tissue we can start implementing kinetic friction
    

    #find maximum force exerted
    if i>120:
        if fe[0] > max_force_exerted[0]:
            max_force_exerted[0] = fe[0]
        if fe[1] > max_force_exerted[1]:
            max_force_exerted[1] = fe[1]

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
        dxh += ddxh*dt -fd

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
        Bones = {'Vertebrae one', 'Vertebrae two', 'Vertebrae three', 'Vertebrae four', 'Vertebrae five', 'Vertebrae six'}
        for collision in collision_dict:
            if collision not in Bones and collision_dict[collision] == True:
                # Set the maximum tissue force, the maximum force exerted by needle pre-puncture
                max_tissue_force = variable_dict[collision]['max_tissue_force']
                #print("Max tissue force: ", max_tissue_force)
                if collision == 'Spinal cord' and i>120:
                    spinal_coord_collision_hit = True
                    spinal_coord_collision = True
                else:
                    spinal_coord_collision = False
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
         255-np.clip(np.linalg.norm(np.abs(K_TISSUE)*(xm-xh)/window_scale)*15,0,255),\
         255-np.clip(np.linalg.norm(np.abs(K_TISSUE)*(xm-xh)/window_scale)*15,0,255)) #if collide else (255, 255, 255)
    
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
    pygame.draw.rect(screenVR,cVerte,wall_layer17, border_radius = 4)

    # Draw all the vertebrae
    screenVR.blit(vertebrae_layer,(vert_rect1[0],vert_rect1[1])) 
    screenVR.blit(vertebrae_layer,(vert_rect2[0],vert_rect2[1]))
    screenVR.blit(vertebrae_layer,(vert_rect3[0],vert_rect3[1]))
    screenVR.blit(vertebrae_layer,(vert_rect4[0],vert_rect4[1]))
    screenVR.blit(vertebrae_layer,(vert_rect5[0],vert_rect5[1]))  
    screenVR.blit(vertebrae_layer,(vert_rect6[0],vert_rect6[1]))
    
    # Draw needle 
    pygame.draw.line(screenVR, cOrange, (haptic.center[0],haptic.center[1]), (haptic.center[0]+np.cos(alpha)*250, haptic.center[1]+ np.sin(alpha)*250), 2 )
    pygame.draw.line(screenVR, cOrange, (haptic.center[0],haptic.center[1]), (haptic.center[0]+np.sin(-alpha)*25, haptic.center[1]+ np.cos(-alpha)*25), 2 )
    pygame.draw.line(screenVR, cOrange, (haptic.center[0],haptic.center[1]), (haptic.center[0]-np.sin(-alpha)*25, haptic.center[1]- np.cos(-alpha)*25), 2 )
    
    
 

    # Indicate drop in needle pressure
    if collision_dict['Cerebrospinal fluid one'] and i > 350:
        text_font = pygame.font.SysFont('Helvetica Neue', 18)
        text_surface = text_font.render('Needle pressure is dropping!', False, (0,0,0))
        screenVR.blit(text_surface, (100, 50))

    #toggle a mask over the spine
    if toggle_visual:
        pygame.draw.rect(screenVR,cSkin,(simulation_space[0][0],0,simulation_space[0][1],simulation_space[1][1]), border_radius = 0)

    if i < 350:
        # # Draw all the vertebrae
        pygame.draw.rect(screenVR,cWhite,(simulation_space[0][0],0,simulation_space[0][1],simulation_space[1][1]), border_radius = 0)

        pygame.draw.rect(screenVR,cVerte,wall_layer12, border_radius = 4)
        pygame.draw.rect(screenVR,cVerte,wall_layer13, border_radius = 4)
        pygame.draw.rect(screenVR,cVerte,wall_layer14, border_radius = 4)
        pygame.draw.rect(screenVR,cVerte,wall_layer15, border_radius = 4)
        pygame.draw.rect(screenVR,cVerte,wall_layer16, border_radius = 4)
        pygame.draw.rect(screenVR,cVerte,wall_layer17, border_radius = 4)

        # Draw all the vertebrae
        screenVR.blit(vertebrae_layer,(vert_rect1[0],vert_rect1[1])) 
        screenVR.blit(vertebrae_layer,(vert_rect2[0],vert_rect2[1]))
        screenVR.blit(vertebrae_layer,(vert_rect3[0],vert_rect3[1]))
        screenVR.blit(vertebrae_layer,(vert_rect4[0],vert_rect4[1]))
        screenVR.blit(vertebrae_layer,(vert_rect5[0],vert_rect5[1]))  
        screenVR.blit(vertebrae_layer,(vert_rect6[0],vert_rect6[1]))

    ##Fuse it back together
    window.blit(screenHaptics, (0,0))
    window.blit(screenVR, (600,0))

    ##Print status in  overlay
    if debugToggle: 
        
        text = font.render("FPS = " + str(round(clock.get_fps())) + \
                            "  xm = " + str(np.round(10*xm)/20000) +\
                            "  xh = " + str(np.round(10*xh)/20000) +\
                            "  fe = " + str(np.round(10*fe)/20000) \
                            , True, (0, 0, 0), (255, 255, 255))
        window.blit(text, textRect)


    if spinal_coord_collision:
        GB = min(255, max(0, round(255 * 0.5)))
        window.fill((255, GB, GB), special_flags = pygame.BLEND_MULT)

    if collision_dict['Cerebrospinal fluid one']:
        GB = min(255, max(0, round(255 * 0.5)))
        window.fill((GB, 255, GB), special_flags = pygame.BLEND_MULT)


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
     ['Distance to fluid: ',(wall_layer6[0] - haptic_endpoint[0])*2,' mm'],
     ['Number of bone hits: ',int(bone_collision_count-1)],
     ['Spinal coord hit: ',spinal_coord_collision_hit],
     ['Maximum exerted force: ',max_force_exerted/10000],
     ['Deviation inside of tissue: ',std_y],
     ['']]
df = pd.DataFrame(data=d)
df.to_csv('test_csv.csv',mode='a',header=False,index=False)