
from tqdm import tqdm
for i in tqdm(range(10)):




    activeEdgesList = []
    activePointsSet = set()
    valuesToCheck = [col1,col2,col3,col4,col5,col6,col7];

    for coordinate in valuesToCheck:
        if coordinate in activeEdgesList:
            activePointsSet.add(coordinate)
            break

    # X = np.load(os.path.join(scriptpath,'hw1.npy'))
    # print(X)

    #
    # yk_min = y + 1; νεες πλευρες
    # yk_max = y εξαιρουμενες
