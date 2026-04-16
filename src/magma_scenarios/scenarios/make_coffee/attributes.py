import sapien

att = {"coffee_pod": ["black", "milky", "white"]}

# mug and capsule positions relative to the coffee maker
dropped_mug_pose = sapien.Pose(p=[-0.22, 0, -0.1], q = [0,1,0,0])
loaded_capsule_pose = sapien.Pose(p=[-0.022, 0.04, 0.18], q = [0,1,0,0])

#team assignement
people = [ "Smith", "Anderson", "Clark", "Wright", "Mitchell", "Johnson", "Thomas", "Rodriguez", "Lopez", "Perez",
    "Williams", "Jackson", "Lewis", "Hill", "Roberts", "Jones", "White", "Lee", "Scott", "Turner", "Brown", "Harris", "Walker"]

teams = ["DISCO","GEPETTO","RAP","RIS","MAC"]