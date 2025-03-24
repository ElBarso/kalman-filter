import os
import numpy as np
import matplotlib.pyplot as plt

# Simple Kalman filter steps:
# 1) k-1 optmized a posteriori estimation is given (opt-post)
# 2) k-1 opt-post --> k|k-1 a priori estimation
# 3) measurement k  is acquired
# 4) Measurement k is used tu optimize the a priori estimate --> k optimized a posteriori estimate
# 5) Repeat
  
def compute_prior_estimates(x_k1, P_k1, F, Q, U):
   """Compute a priori estimates for both state and state covariance."""
   x_k_k1 = x_k_k1 = F @ x_k1 + U
   P_k_k1 = P_k_k1 = F @ P_k1 @ F.T + Q
   return x_k_k1, P_k_k1

def compute_kalman_gain(P, H, R):
   """Compute Kalman Gain."""
   return P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

def update_prior_estimates(state_prior_estimate, state_covariance_prior_estimate, measurement, H, R):
   """Given the current iteration measurement, 
   update both a priori estimates to a posteriori optimized estimates  
   for state and state covariance."""
   kalman_gain = compute_kalman_gain(P=state_covariance_prior_estimate, H=H, R=R)
   updated_state = state_prior_estimate + kalman_gain @ (measurement - H@state_prior_estimate)
   updated_covariance_state = (np.eye(4) - kalman_gain@H) @ state_covariance_prior_estimate
   return updated_state, updated_covariance_state

def generate_2D_bullet_trajectory(x_0, y_0, vx_0, vy_0, delta_t, G, Q,  total_duration=100):
   """Generate x and y coordinates for a parabolic 2D motion in Earth's gravitational field."""
   
   # Initial conditions
   x_n1  = x_0
   y_n1  = y_0
   vx_n1 = vx_0
   vy_n1 = vy_0
   
   # Initialize trajectory points list
   trajectory = []
   
   # Set thenumber of points to compute 
   # and compute them according to parabolic motion cinetic equations
   steps = int(np.ceil(total_duration / delta_t) + 1)
   if steps > 1:
      for current_step, time_step in enumerate(np.linspace(0,1, steps)):
         if y_n1 >= 0:
            noise = 50*np.random.randn()
            # w = np.linalg.cholesky(Q)*np.random.normal(0, 1, 4)
            x_n = x_n1 + vx_n1*delta_t
            # y_n = y_n1 + vy_n1*delta_t  - (G/2)*delta_t**2 + w[2,2]
            y_n = y_n1 + vy_n1*delta_t  - (G/2)*delta_t**2 + noise
            vx_n = vx_n1
            vy_n = vy_n1 - G*delta_t
            
            # Store generate point
            trajectory.append((x_n1, y_n1))
            
            # Update variables for next iteration
            x_n1 = x_n
            y_n1 = y_n
            vx_n1 = vx_n
            vy_n1 = vy_n
         
         else:
            steps = current_step
            break
   
   return np.array(trajectory), steps
            
###################################################################################     

def implement_kalman_filter(X_0, P_0, F, H, U, Q, R, G, measurements, time_steps):
   
   filtered_trajectory = []
   filtered_velocities = []   
      
   # set intial conditions
   x_k1 = X_0
   P_k1 = P_0
   
   # For each 
   for i in np.arange(time_steps):
      x_k_k1, P_k_k1 = compute_prior_estimates(x_k1, P_k1, F=F, Q=Q, U=U)
      there_are_no_missing_measures = not np.isnan(measurements[i]).all()
      if there_are_no_missing_measures:
         updated_state, updated_covariance_state = update_prior_estimates(state_prior_estimate=x_k_k1, 
                                                                        state_covariance_prior_estimate=P_k_k1, 
                                                                        measurement=measurements[i], 
                                                                        H=H, 
                                                                        R=R)   
         # rename variables for clarity
         x_k1 = updated_state
         P_k1 = updated_covariance_state
         
         filtered_trajectory.append((updated_state[0], updated_state[1]))
         filtered_velocities.append((updated_state[2], updated_state[3]))
      else:         
         x_k1 = x_k_k1
         P_k1 = P_k_k1
         filtered_trajectory.append((x_k_k1[0], x_k_k1[1]))
         filtered_velocities.append((x_k_k1[2], x_k_k1[3]))
      
      
   return np.array(filtered_trajectory), np.array(filtered_velocities)
         
      
if __name__ == "__main__":
   output_folder_path = "outputs"
   
   G = 9.81
   DELTA_T = 1

   F = np.array([
                  [1, 0, DELTA_T, 0], 
                  [0, 1, 0, DELTA_T],
                  [0, 0, 1, 0],              
                  [0, 0, 0, 1]
               ])
   
   U = [
      np.array([0,0,0,+G*DELTA_T]), 
        np.array([0,0,0,0]),
        np.array([0,0,0,-G*DELTA_T])
        ]

   H = np.array([
                  [1, 0, 0, 0],
                  [0, 1, 0, 0]
               ])

   Q = 0.1*np.eye(4)
   R = 500*np.eye(2)
   P_0 = 1e6*Q
   X_0 = np.array([0, 0, 300, 600])
   
   # Plot generated trajectories
   fig = plt.figure()
   x_limits = []
   for u in U:
      experiment_traj, time_steps=generate_2D_bullet_trajectory(*X_0, DELTA_T, G=G, Q=Q, total_duration=150)
      kalman_traj, kalman_velocities = implement_kalman_filter(X_0, P_0, F, H, u, Q, R, G, experiment_traj, time_steps)
      print("{kal} {exp}".format(kal=kalman_traj.shape, exp=experiment_traj.shape))
      plt.plot(kalman_traj[:,0], kalman_traj[:,1], label=f'u={u}',lw=1)
   plt.plot(experiment_traj[:,0], experiment_traj[:,1], label="experiment")
   
   plt.legend()
   plt.savefig(os.path.join(output_folder_path, 'trajectory.png'))
   plt.clf()
   
   # Plot and save x trajecotory of simulated 
   plt.figure()
   plt.plot(experiment_traj[:,0], kalman_traj[:,0])
   plt.savefig(os.path.join(output_folder_path, 'x_relation_check.png'))   
   
   
