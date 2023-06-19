function mapRE(re_list, map_matrix)
    result = fill(NaN, size(map_matrix)[1], size(map_matrix)[2])
    for i in 1:length(re_list)
        result[:, i] = @views re_list[i][map_matrix[:, i]]
    end
    return result
end

function _count_β(β)
    return size(β, 1) <= 2 ? 0 : size(β, 1) - 2
end