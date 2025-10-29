import numpy as np

def role_assignment(teammate_positions, formation_positions):

    # Input : Locations of all teammate locations and positions
    # Output : Map from unum -> positions
    #-----------------------------------------------------------#
    
    teammates = [tuple(p) for p in teammate_positions]
    formations = [tuple(p) for p in formation_positions]

    N = len(teammates)
    assert N == len(formations), "Number of teammates and formation positions must be equal"

   
    dist = np.zeros((N, N), dtype=float)
    for p in range(N):
        for f in range(N):
            dx = teammates[p][0] - formations[f][0]
            dy = teammates[p][1] - formations[f][1]
            dist[p, f] = (dx*dx + dy*dy) ** 0.5

    # Player Preference
    players_pref = [list(np.argsort(dist[p, :])) for p in range(N)]

    # positions_pref[f] = list of player indices sorted by increasing distance (best first)
    positions_pref = [list(np.argsort(dist[:, f])) for f in range(N)]

   
    pos_rank = [None] * N
    for f in range(N):
        rank = {}
        for rank_idx, p in enumerate(positions_pref[f]):
            rank[p] = rank_idx
        pos_rank[f] = rank

    # Gale-Shapley where players "propose" to positions
    free_players = list(range(N))              
    next_proposal_index = [0] * N              
    matched_position = [-1] * N                
    matched_player = [-1] * N                  

    while free_players:
        p = free_players.pop(0)
        if next_proposal_index[p] >= N:
            continue
        f = players_pref[p][next_proposal_index[p]]
        next_proposal_index[p] += 1

        current = matched_position[f]
        if current == -1:
            matched_position[f] = p
            matched_player[p] = f
        else:
            if pos_rank[f][p] < pos_rank[f][current]:
                matched_position[f] = p
                matched_player[p] = f
                matched_player[current] = -1
                free_players.append(current)
            else:
                free_players.append(p)

    point_preferences = {}
    for p in range(N):
        f = matched_player[p]
        point_preferences[p + 1] = formations[f]

    return point_preferences