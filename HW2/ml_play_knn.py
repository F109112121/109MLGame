
"""The template of the main script of the machine learning process
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

def ml_loop():
    """The main loop of the machine learning process
    This loop is run in a separate process, and communicates with the game process.
    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.
    import pickle
    import numpy as np
    filename="C:\\Users\\user\\Desktop\\MLGame-beta4\\knn_example1.sav"  #knn_example1.sav 路徑
    model = pickle.load(open(filename, 'rb'))
    #print(model)
    ball_position_history=[]
    vx = 0
    vy = 0
    ball_destination = 0
    ball_going_down = 0
    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()

    # 3. Start an endless loop.
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()
        ball_position_history.append(scene_info.ball)
        platform_center_x = scene_info.platform[0] +20 #platform length = 40

        if (len(ball_position_history))==1:
            platform_ok= 0
            ball_going_down=0
        elif ball_position_history[-1][1]-ball_position_history[-2][1] > 0:
            ball_going_down = 1
            vy=ball_position_history[-1][1]-ball_position_history[-2][1]
            vx=ball_position_history[-1][0]-ball_position_history[-2][0]
        else:
            ball_going_down = 0
            platform_ok= 0
        
        
        if scene_info.status == GameStatus.GAME_OVER or \
            scene_info.status == GameStatus.GAME_PASS:
            comm.ml_ready()        
            continue
        
        
        
        if ball_going_down == 1 and ball_position_history[-1][1] >= 0:
            ball_destination = ball_position_history[-1][0]+((395-ball_position_history[-1][1])/vy)*vx
            if ball_destination >= 195:
                ball_destination = 195-(ball_destination-195)
            elif ball_destination <= 0:
                ball_destination = - ball_destination
        else:
            ball_destination = platform_center_x
        
        if(len(ball_position_history) > 2):
            vx1=ball_position_history[-1][0]
            vx2=ball_position_history[-2][0]
            vy1=ball_position_history[-1][1]
            vy2=ball_position_history[-2][1]
            inp_temp=np.array([vx2, vy2, vx1, vy1])
            int_temp=np.array([scene_info.platform[0]])
            input=inp_temp[np.newaxis, :]
            input1=int_temp[np.newaxis, :]
            print("球的x , y位置 " ,end='\n')
            print(input)
            print("板子的位置" ,end='\n')
            print(input1)
            
        if platform_center_x < ball_destination-10:
            comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
        elif platform_center_x > ball_destination+10:  
            comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
            
