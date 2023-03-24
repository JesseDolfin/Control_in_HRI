def bone_colision(p,collision_bone,pr,flag):
    if collision_bone and flag:
        phold = p
        flag = False
       

    if pr[0] > phold[0] and not flag:
         ddp = 0
    else:
        flag = True