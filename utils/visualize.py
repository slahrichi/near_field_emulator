import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_fields(fields, title, cmap='viridis', save_path=None, frames=63, interval=100):
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
    ax.set_title(f'{title} - Frame 0/{frames}')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # frame updating
    def update(frame):
        im.set_array(fields[:, :, frame])
        ax.set_title(f'{title} - Frame {frame}/{frames}')
        return [im]
    
    # Create: Animation object
    anim = FuncAnimation(
        fig, 
        update,
        frames=frames,
        interval=interval, # ms between frames 
        blit=True
    )
    
    if save_path is not None:
        anim.save(save_path)
        plt.close() 
    else:
        return anim
