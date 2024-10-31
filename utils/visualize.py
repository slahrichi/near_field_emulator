import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_fields(fields, title, cmap='viridis', save_path=None):
    """
    Create an animation of 2D field slices over time.
    
    Args:
        fields (np.ndarray): Array of shape (x, y, t) containing field values
        title (str): Title for the animation
        cmap (str): Colormap to use for visualization
        
    Returns:
        matplotlib.animation.FuncAnimation: Animation object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Initialize: Frame 0
    im = ax.imshow(fields[:, :, 0], cmap=cmap, animated=True)
    ax.set_title(f'{title} - Frame 0/63')
    
    # frame updating
    def update(frame):
        im.set_array(fields[:, :, frame])
        ax.set_title(f'{title} - Frame {frame}/63')
        return [im]
    
    # Create: Animation object
    anim = FuncAnimation(
        fig, 
        update,
        frames=63,
        interval=100, # ms between frames 
        blit=True
    )
    
    plt.close()  # Prevent display of static plot
    
    if save_path is not None:
        anim.save(save_path)
        
    return anim
