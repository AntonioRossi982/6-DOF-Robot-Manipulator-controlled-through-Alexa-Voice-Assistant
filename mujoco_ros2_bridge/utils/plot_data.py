import matplotlib.pyplot as plt
import numpy as np

def plot_error_history(error_history, dt, title="PD Controller Errors"):
    """
    Plots the position error over time.
    
    Args:
        error_history: list of dictionaries containing ‘position_error’ 
                       from the PDController history
        dt: simulation time step [s]
        title: plot title
    """
    # check if error_history is empty
    if not error_history:
        print('No error data to plot')
        return

    # Extract errors from the history
    errors = np.array([
        entry['position_error']
        for entry in error_history
    ])

    # Create the time axis
    time_axis = np.arange(len(errors)) * dt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(title)

    # Joint 1
    axes[0].plot(time_axis, errors[:, 0], 'b-', linewidth=1.5)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Joint 1 Error [rad]')
    axes[0].set_xlabel('Time [s]')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Joint 1 Position Error')

    # Joint 2
    axes[1].plot(time_axis, errors[:, 1], 'g-', linewidth=1.5)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Joint 2 Error [rad]')
    axes[1].set_xlabel('Time [s]')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Joint 2 Position Error')

    plt.tight_layout()
    plt.savefig('/tmp/pd_errors.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Graph saved to /tmp/pd_errors.png")