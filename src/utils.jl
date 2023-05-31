function mapRE(re_list, map_matrix)
    result = fill(NaN, size(map_matrix)[1], size(map_matrix)[2])
    for i in 1:length(re_list)
        result[:, i] = @views re_list[i][map_matrix[:, i]]
    end
    return result
end