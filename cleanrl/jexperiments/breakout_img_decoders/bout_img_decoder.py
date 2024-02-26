
#Action VALUES
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3

ERROR_MARGIN = 3
"""Error margin for the ball position. If the ball is within this margin, the bar will not move"""

def get_human_action(env_obs, img_size):
    """
    Returns an action that directs the bar to the ball position
    It takes into account the ball speed. If speed in None (happend when ball is not found or
    in the first frame), follows the ball position. If ball position is not known, the FIRE action is chosen
    """
    
    if img_size == 64:
        from img64 import get_bar_pos, get_ball_pos, get_ball_speed
    elif img_size == 84:
        from img84 import get_bar_pos, get_ball_pos, get_ball_speed
    elif img_size == 128:
        from img128 import get_bar_pos, get_ball_pos, get_ball_speed
    else:
        raise ValueError("img_size must be 64, 84 or 128")
    
    
    bar_pos = get_bar_pos(env_obs[-1])
    ball_pos = get_ball_pos(env_obs[-1])
    ball_speed = get_ball_speed(env_obs[-2],env_obs[-1],)
    
    if ball_pos[0] < 0 and ball_pos[0] < 0:
        # Ball is not known in this frame
        return FIRE
    
    future_col = ball_pos[0] + ball_speed[0]
    
    if future_col + ERROR_MARGIN > bar_pos:
        return RIGHT
    elif future_col - ERROR_MARGIN < bar_pos:
        return LEFT
    else:
        return NOOP