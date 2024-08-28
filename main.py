import numpy as np

def main():
    test_construct_person_hand_match_matrix_when_p_is_smaller_than_2hc()
    test_construct_person_hand_match_matrix_when_p_is_greater_than_2hc()

def construct_person_hand_match_matrix_when_p_is_smaller_than_2hc(distance_matrix):
    match_matrix = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]), dtype=bool)

    for i, row in enumerate(distance_matrix):
        indices_sorted_in_ascending_order = np.argsort(row)
        for index in indices_sorted_in_ascending_order:
            if np.sum(match_matrix[:, index]) == False:
                match_matrix[i, index] = True
                break

    col_indices_where_false = []
    for i in range(match_matrix.shape[1]):
        if np.sum(match_matrix[:, i]) == False:
            col_indices_where_false.append(i)

    new_distance_matrix = distance_matrix[:, col_indices_where_false]

    person_index_not_occupied = [i for i in range(match_matrix.shape[0])]
    while col_indices_where_false and person_index_not_occupied:
        row_indices, col_indices = np.where(new_distance_matrix == np.min(new_distance_matrix))
        smallest_row_index, smallest_col_index = row_indices[0], col_indices[0]
        match_matrix[person_index_not_occupied[smallest_row_index], col_indices_where_false[smallest_col_index]] = True
        col_indices_where_false.pop(smallest_col_index)
        person_index_not_occupied.pop(smallest_row_index)
        new_distance_matrix = np.delete(new_distance_matrix, smallest_row_index, axis=0)
        new_distance_matrix = np.delete(new_distance_matrix, smallest_col_index, axis=1)
        
    return match_matrix


def construct_person_hand_match_matrix_when_p_is_greater_than_2hc(distance_matrix):
    match_matrix = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]), dtype=bool)

    for col_index in range(distance_matrix.shape[1]):
        column = distance_matrix[:, col_index]
        smallest_index = np.argpartition(column, 0)[0]
        match_matrix[smallest_index, col_index] = True

    column_for_new_distance_matrix = []
    row_for_new_distance_matrix = []
    for row_index in range(distance_matrix.shape[0]):
        if np.sum(match_matrix[row_index]) >2:
            indices_where_false= np.where(match_matrix[row_index] == False)
            col_indices_sorted_in_ascending_order = np.argsort(distance_matrix[row_index])
            col_indices_where_true_sorted_in_ascending_order = np.delete(col_indices_sorted_in_ascending_order, indices_where_false)[2:]
            match_matrix[row_index, col_indices_where_true_sorted_in_ascending_order] = False
            column_for_new_distance_matrix.extend(col_indices_where_true_sorted_in_ascending_order)
        else:
            row_for_new_distance_matrix.append(row_index)

    new_distance_matrix = distance_matrix[row_for_new_distance_matrix, :][:, column_for_new_distance_matrix]
    
    while column_for_new_distance_matrix:
        row_indices, col_indices = np.where(new_distance_matrix == np.min(new_distance_matrix))
        smallest_row_index, smallest_col_index = row_indices[0], col_indices[0]
        match_matrix[row_for_new_distance_matrix[smallest_row_index], column_for_new_distance_matrix[smallest_col_index]] = True
        column_for_new_distance_matrix.pop(smallest_col_index)
        new_distance_matrix = np.delete(new_distance_matrix, smallest_col_index, axis=1)
        
        if np.sum(match_matrix[smallest_row_index, :]) >= 2:
            row_for_new_distance_matrix.pop(smallest_row_index)
            new_distance_matrix = np.delete(new_distance_matrix, smallest_row_index, axis=0)
        
    return match_matrix

def test_output(match_matrix):
    for i in range(match_matrix.shape[1]):
        if np.sum(match_matrix[:, i]) == 2:
            print("Test Failed")
            return
    print("Test Passed")

def test_construct_person_hand_match_matrix_when_p_is_smaller_than_2hc():
    # Test Case 1: Small 2x4 matrix
    distance_matrix_1 = np.array([
        [1, 2, 10, 10],
        [10, 10, 2, 1]
    ])

    print("Test Case 1: construct_person_hand_match_matrix_when_p_is_smaller_than_2hc")
    match_matrix_1 = construct_person_hand_match_matrix_when_p_is_smaller_than_2hc(distance_matrix_1)
    test_output(match_matrix_1)


    # Test Case 2: Larger 2x5 matrix with distinct values
    distance_matrix_2 = np.array([
        [1, 2, 10, 10, 20],
        [10, 10, 2, 1, 20]
    ])

    print("Test Case 2: construct_person_hand_match_matrix_when_p_is_smaller_than_2hc")
    match_matrix_2 = construct_person_hand_match_matrix_when_p_is_smaller_than_2hc(distance_matrix_2)
    test_output(match_matrix_2)


    # Test Case 3: Edge case, single row
    distance_matrix_3 = np.array([
        [1, 3, 1]
    ])

    print("Test Case 3: construct_person_hand_match_matrix_when_p_is_smaller_than_2hc")
    match_matrix_3 = construct_person_hand_match_matrix_when_p_is_smaller_than_2hc(distance_matrix_3)
    test_output(match_matrix_3)

def test_construct_person_hand_match_matrix_when_p_is_greater_than_2hc():
    # Test Case 1: Small 2x4 matrix
    distance_matrix_1 = np.array([
        [1, 2, 10, 10],
        [10, 10, 2, 1]
    ])

    print("Test Case 1: construct_person_hand_match_matrix_when_p_is_smaller_than_2hc")
    match_matrix_1 = construct_person_hand_match_matrix_when_p_is_greater_than_2hc(distance_matrix_1)
    test_output(match_matrix_1)


    # Test Case 2: Larger 3x5 matrix with distinct values
    distance_matrix_2 = np.array([
        [1, 2, 10, 10, 20],
        [10, 10, 2, 1, 20],
        [11, 11, 11, 11, 1]
    ])

    print("Test Case 2: construct_person_hand_match_matrix_when_p_is_smaller_than_2hc")
    match_matrix_2 = construct_person_hand_match_matrix_when_p_is_greater_than_2hc(distance_matrix_2)
    test_output(match_matrix_2)


    # Test Case 3: 3x5 matrix 
    distance_matrix_3 = np.array([
        [1, 2, 10, 3, 20],
        [10, 3, 1, 10, 20],
        [11, 11, 11, 11, 1]
    ])

    print("Test Case 3: construct_person_hand_match_matrix_when_p_is_smaller_than_2hc")
    match_matrix_3 = construct_person_hand_match_matrix_when_p_is_greater_than_2hc(distance_matrix_3)
    test_output(match_matrix_3)


    # Test Case 4: 3x4 matrix 
    distance_matrix_4 = np.array([
        [1, 2, 10, 3],
        [10, 12, 1, 2],
        [11, 11, 11, 11]
    ])

    print("Test Case 4: construct_person_hand_match_matrix_when_p_is_smaller_than_2hc")
    match_matrix_4 = construct_person_hand_match_matrix_when_p_is_greater_than_2hc(distance_matrix_4)
    test_output(match_matrix_4)


    # Test Case 5: 3x3 matrix 
    distance_matrix_5 = np.array([
        [1, 3, 2],
        [10, 5, 9],
        [11, 11, 11]
    ])

    print("Test Case 5: construct_person_hand_match_matrix_when_p_is_smaller_than_2hc")
    match_matrix_5 = construct_person_hand_match_matrix_when_p_is_greater_than_2hc(distance_matrix_5)
    test_output(match_matrix_5)


    # Test Case 6: 3x2 matrix 
    distance_matrix_6 = np.array([
        [1, 12],
        [10, 10],
        [1, 5]
    ])

    print("Test Case 6: construct_person_hand_match_matrix_when_p_is_smaller_than_2hc")
    match_matrix_6 = construct_person_hand_match_matrix_when_p_is_greater_than_2hc(distance_matrix_6)
    test_output(match_matrix_6)

if __name__ == "__main__":
    main()