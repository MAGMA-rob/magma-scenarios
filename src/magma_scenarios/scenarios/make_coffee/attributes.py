import sapien, random

att = {"coffee_pod": ["black", "milky", "white"]}

# mug and capsule positions relative to the coffee maker
dropped_mug_pose = sapien.Pose(p=[-0.22, 0, -0.1], q = [0,1,0,0])
loaded_capsule_pose = sapien.Pose(p=[-0.022, 0.04, 0.18], q = [0,1,0,0])

#team assignement
people = [ "Smith", "Anderson", "Clark", "Wright", 
        "Mitchell", "Johnson", "Thomas", "Rodriguez", 
        "Lopez", "Perez", "Williams", "Jackson",
        "Lewis", "Hill", "Roberts", "Jones", "White",
        "Lee", "Scott", "Turner", "Brown", "Harris", 
        "Walker", "Arthur", "Matthieu", "Florent", 
        "Ariane", "Abdelbasset", "Michel", "Olivier",
        "Philippe", "Hector", "Michael", "Leo",
        "Marta", "Alexis", "Zhiang", "Donald",
        "Tim", "Theo", "Emma", "Camille", "Solene",
        "Alexandra", "Angela", "Merlin", "Yanis"]

teams = ["DISCO","GEPETTO","RAP","RIS","MAC"]


def build_people_assignment(nb_team : int, nb_people_per_team : int):
        people_copy = people.copy()
        teams_copy = teams.copy()

        random.shuffle(people_copy)
        random.shuffle(teams_copy)

        if nb_team > len(teams_copy):
                raise TypeError(f"Only {len(teams_copy)} exists but you asked for {nb_team}")
        if nb_people_per_team * nb_team > len(people_copy):
                raise TypeError(f"You asked for {nb_people_per_team} for {nb_team} but only {len(people_copy)} \
                                people names exists ({nb_people_per_team*nb_team})")

        selected_people = people_copy[:nb_people_per_team*nb_team]
        teams_selected = teams_copy[:nb_team]

        team_dict = {}

        for i,team in enumerate(teams_selected) :
                start = i*nb_people_per_team
                end = (i+1)*nb_people_per_team
                team_dict[team] = selected_people[start:end]

        return team_dict