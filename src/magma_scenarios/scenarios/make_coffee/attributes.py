import sapien

att = {"coffee_pod": ["black", "milky", "white"]}

# mug and capsule positions relative to the coffee maker
dropped_mug_pose = sapien.Pose(p=[-0.22, 0, -0.1], q = [0,1,0,0])
loaded_capsule_pose = sapien.Pose(p=[-0.022, 0.04, 0.18], q = [0,1,0,0])