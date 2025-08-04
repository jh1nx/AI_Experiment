def ResolutionProp(KB):
    result = []     #用于存储结果
    clauses = list(KB) #将KB中的子句存入列表clauses中
    
    for i, clause in enumerate(clauses, 1):
        result.append(f"{i} {format_clause(clause, i)}")
    
    existing_clauses = set(clauses) #存储已有的子句,避免重复
    used_clauses_idx = set()  
    
    new_clause_idx = len(clauses) + 1  
    
    old_idx = 0  
    new_idx = len(clauses)  #新句子的起始索引
    while old_idx < len(clauses):
        new_clause_added = False    #判读是否有新子句
        
        #遍历子句对
        for i in range(new_idx):
            if i in used_clauses_idx:
                continue  
                
            for j in range(i+1, len(clauses)):
                if j in used_clauses_idx:
                    continue  
                
                resolvent_found, new_clause, i_part, j_part = resolve(clauses[i], clauses[j])
                
                if resolvent_found and new_clause not in existing_clauses:
                    clauses.append(new_clause)
                    existing_clauses.add(new_clause)
                    
                    used_clauses_idx.add(i)
                    used_clauses_idx.add(j)
                    
                    #添加到结果中
                    step = f"{new_clause_idx} R[{i+1}{i_part},{j+1}{j_part}]={format_clause(new_clause)}"
                    result.append(step)
                    
                    new_clause_idx += 1
                    new_clause_added = True
                    
                    #结束循环
                    if new_clause == ():
                        return result
        
        if not new_clause_added:
            break
            
        old_idx = new_idx
        new_idx = len(clauses)
    
    return result

#这个函数的作用是解析子句
def resolve(clause1, clause2):
    for i, lit1 in enumerate(clause1):
        for j, lit2 in enumerate(clause2):
            if (lit1.startswith('~') and lit1[1:] == lit2) or (lit2.startswith('~') and lit2[1:] == lit1):
                new_clause_list = []
                for lit in clause1:
                    if lit != lit1:
                        new_clause_list.append(lit)
                for lit in clause2:
                    if lit != lit2:
                        new_clause_list.append(lit)
                
                #对新子句进行排序和去重
                new_clause = tuple(sorted(set(new_clause_list)))
                
                i_part = get_literal_marker(clause1, i)
                j_part = get_literal_marker(clause2, j)
                
                return True, new_clause, i_part, j_part
    
    return False, (), "", ""

#这个函数的作用是格式化子句
def format_clause(clause, clause_idx=0):
    if not clause:
        return "()"
    
    if len(clause) == 1:
        return f"({clause[0]},)"
    
    result = "("
    for i, lit in enumerate(clause):
        if i > 0:
            result += ","
        result += lit  
    result += ")"
    
    return result

#这个函数的作用是获取文字标记（即a、b、c……）
def get_literal_marker(clause, idx):
    """获取文字标记"""
    if len(clause) <= 1:
        return ""  
    return chr(97 + idx)  # a, b, c, ...

if __name__ == "__main__":
    KB = [("FirstGrade",), ("~FirstGrade", "Child"), ("~Child",)]
    result = ResolutionProp(KB)
    for step in result:
        print(step) #输出结果
