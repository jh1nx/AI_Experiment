from copy import deepcopy

class FirstOrderLogic:      #初始化一阶逻辑原子公式
    def __init__(self, is_negated, predicate, arguments):
        self.is_negated = is_negated #是否被否定
        self.predicate = predicate #谓词名称
        self.arguments = arguments #谓词参数列表
    def __str__(self):     #将一阶逻辑原子公式转换为字符串表示
        return ("" if self.is_negated else "~") + self.predicate + '(' + list_to_str(self.arguments) + ')'
    
class ResolutionStep: #初始化归结步骤
    def __init__(self, is_premise: bool, mgu_dict: dict, parent1_index: int, parent1_atom_index: int, parent2_index: int, parent2_atom_index: int, result_clause=list[FirstOrderLogic]):
        self.mgu_dict = mgu_dict    #最一般合一（MGU）字典
        self.parent1_index = parent1_index + 1  #父句的索引
        self.parent2_index = parent2_index + 1  
        self.is_premise = is_premise    #是否是前提（初始子句）
        self.parent1_atom_index = parent1_atom_index    #父句中归结原子的索引
        self.parent2_atom_index = parent2_atom_index
        self.result_clause = result_clause  #归结后的新子句
    
    def __str__(self):  #将归结步骤转换为字符串表示
        if self.is_premise:
            return f"({FOL_list_to_str(self.result_clause)[:-1]})"
        else:
            mgu_str = "("
            for key, value in self.mgu_dict.items():
                mgu_str += key + "=" + value + " "
            mgu_str = mgu_str[:-1] + ")" if len(self.mgu_dict) > 0 else ""
            result_str = FOL_list_to_str(self.result_clause)[:-1]
            return f"R[{self.parent1_index}{num_to_letter(self.parent1_atom_index) if self.parent1_atom_index >= 0 else ''},{self.parent2_index}{num_to_letter(self.parent2_atom_index) if self.parent2_atom_index >= 0 else ''}]{mgu_str} = ({result_str})"

def convert_list_to_FOL(clause_list):   #将字符串列表转换为一阶逻辑原子公式列表
    FOL_list = []
    for clause in clause_list:
        FOL_clause = []
        for atom in clause:
            is_negated = (atom[0] != '~')
            start_bracket = atom.index('(')
            end_bracket = atom.index(')')
            arguments = atom[start_bracket+1:end_bracket].split(',')
            predicate = atom[:start_bracket] if is_negated else atom[1:start_bracket]
            FOL_clause.append(FirstOrderLogic(is_negated, predicate, arguments))
        FOL_list.append(FOL_clause)
    return FOL_list

def FOL_list_to_str(FOL_list):  #将一阶逻辑原子公式列表转换为字符串表示
    result_str = ""
    for clause in FOL_list:
        result_str += str(clause) + ","
    return result_str

def is_clause_not_in_kb(clause: list[FirstOrderLogic], knowledge_base: list[list[FirstOrderLogic]]):    #检查子句是否不在知识库中
    for kb_clause in knowledge_base:
        if clause == kb_clause:
            return False
    return True

def resolution_algorithm(knowledge_base: list[list[FirstOrderLogic]]):  #实现归结算法
    def perform_resolution(clause1: list[FirstOrderLogic], clause2: list[FirstOrderLogic], clause1_index: int, clause2_index: int, knowledge_base: list[list[FirstOrderLogic]], resolution_steps: list[ResolutionStep]):
        for i in range(len(clause1)):
            for j in range(len(clause2)):
                if clause1[i].is_negated != clause2[j].is_negated and clause1[i].predicate == clause2[j].predicate:
                    if clause1[i].arguments == clause2[j].arguments:
                        if is_clause_not_in_kb(clause1[:i] + clause1[i+1:] + clause2[:j] + clause2[j+1:], knowledge_base):  
                            mgu_dict = {}
                            knowledge_base.append(clause1[:i] + clause1[i+1:] + clause2[:j] + clause2[j+1:])
                            if len(clause1) == 1 and len(clause2) == 1:
                                resolution_steps.append(ResolutionStep(False, mgu_dict, clause1_index, -1, clause2_index, -1, knowledge_base[-1]))
                            elif len(clause1) == 1 and len(clause2) > 1:
                                resolution_steps.append(ResolutionStep(False, mgu_dict, clause1_index, -1, clause2_index, j, knowledge_base[-1]))
                            elif len(clause1) > 1 and len(clause2) == 1:
                                resolution_steps.append(ResolutionStep(False, mgu_dict, clause1_index, i, clause2_index, -1, knowledge_base[-1]))
                            elif len(clause1) > 1 and len(clause2) > 1:
                                resolution_steps.append(ResolutionStep(False, mgu_dict, clause1_index, i, clause2_index, j, knowledge_base[-1]))
                            if len(knowledge_base[-1]) == 0:
                                return True
                    elif clause1[i].arguments != clause2[j].arguments:
                        mgu_dict = {}
                        can_unify = True
                        clause1_copy = deepcopy(clause1)
                        clause2_copy = deepcopy(clause2)
                        
                        for k in range(len(clause1[i].arguments)):
                            if len(clause1[i].arguments[k]) > len(clause2[j].arguments[k]):
                                mgu_dict[clause2[j].arguments[k]] = clause1[i].arguments[k]
                            elif len(clause1[i].arguments[k]) < len(clause2[j].arguments[k]):
                                mgu_dict[clause1[i].arguments[k]] = clause2[j].arguments[k]
                            elif clause1[i].arguments[k] != clause2[j].arguments[k]:
                                can_unify = False
                                break
                        
                        if can_unify:
                            for var, val in mgu_dict.items():
                                for l in range(len(clause1_copy)):
                                    for m in range(len(clause1_copy[l].arguments)):
                                        if clause1_copy[l].arguments[m] == var:
                                            clause1_copy[l].arguments[m] = val
                                for l in range(len(clause2_copy)):
                                    for m in range(len(clause2_copy[l].arguments)):
                                        if clause2_copy[l].arguments[m] == var:
                                            clause2_copy[l].arguments[m] = val
                            
                            result = clause1_copy[:i] + clause1_copy[i+1:] + clause2_copy[:j] + clause2_copy[j+1:]
                            if is_clause_not_in_kb(result, knowledge_base):
                                knowledge_base.append(result)
                                if len(clause1) == 1 and len(clause2) == 1:
                                    resolution_steps.append(ResolutionStep(False, mgu_dict, clause1_index, -1, clause2_index, -1, knowledge_base[-1]))
                                elif len(clause1) == 1 and len(clause2) > 1:
                                    resolution_steps.append(ResolutionStep(False, mgu_dict, clause1_index, -1, clause2_index, j, knowledge_base[-1]))
                                elif len(clause1) > 1 and len(clause2) == 1:
                                    resolution_steps.append(ResolutionStep(False, mgu_dict, clause1_index, i, clause2_index, -1, knowledge_base[-1]))
                                elif len(clause1) > 1 and len(clause2) > 1:
                                    resolution_steps.append(ResolutionStep(False, mgu_dict, clause1_index, i, clause2_index, j, knowledge_base[-1]))
                                if len(knowledge_base[-1]) == 0:
                                    return True
        return False

    resolution_steps = []
    for i in range(len(knowledge_base)):
        resolution_steps.append(ResolutionStep(True, {}, -1, -1, -1, -1, knowledge_base[i]))
    for i in range(len(knowledge_base)):
        for j in range(len(knowledge_base)):
            for k in range(i+1, len(knowledge_base)):
                if perform_resolution(knowledge_base[j], knowledge_base[k], j, k, knowledge_base, resolution_steps):
                    return resolution_steps
    return None

def remove_unused_steps(resolution_steps: list[ResolutionStep], index: int) -> list[ResolutionStep]:    #通过深度优先搜索删除未使用的归结步骤rn: 清理后的归结步骤列表
    used = [False] * len(resolution_steps)

    index_map = {}

    def mark_used_steps(idx):
        if used[idx] or resolution_steps[idx].is_premise:
            return
        used[idx] = True
 
        if not resolution_steps[idx].is_premise:
            mark_used_steps(resolution_steps[idx].parent1_index - 1)
            mark_used_steps(resolution_steps[idx].parent2_index - 1)

    mark_used_steps(index)
    
    for i in range(len(resolution_steps)):
        if resolution_steps[i].is_premise:
            used[i] = True

    cleaned_steps = []
    new_index = 1
    for i in range(len(resolution_steps)):
        if used[i]:
            index_map[i+1] = new_index
            cleaned_steps.append(resolution_steps[i])
            new_index += 1

    for i in range(len(cleaned_steps)):
        if not cleaned_steps[i].is_premise:
            cleaned_steps[i].parent1_index = index_map[cleaned_steps[i].parent1_index]
            cleaned_steps[i].parent2_index = index_map[cleaned_steps[i].parent2_index]
    
    return cleaned_steps

def are_opposites(a: str, b: str) -> bool:  #检查两个字符串是否互为否定形式
    if "~" + a == b or "~" + b == a:
        return True
    return False

def num_to_letter(num):     #将数字转换为对应的小写字母
    return chr(ord('a') + num)

def list_to_str(lst):       #将列表转换为逗号分隔的字符串
    return ",".join(lst)

def extract_digits(s: str) -> int:  #从字符串中提取数字并返回
    digits = ''.join([c for c in s if c.isdigit()])
    return digits if digits else ["0","0"]

def parse_input_string(input_str):  #将输入字符串解析为子句列表
    result = []

    paren_count = 0
    current_clause = ""
    
    i = 0
    while i < len(input_str):
        c = input_str[i]
        
        if c == '(':
            paren_count += 1
            current_clause += c
        elif c == ')':
            paren_count -= 1
            current_clause += c

            if paren_count == 0 and current_clause.startswith('('):
                inner_str = current_clause[1:-1]
                inner_parts = []
                
                inner_paren_count = 0
                inner_part = ""
                
                for j in range(len(inner_str)):
                    ch = inner_str[j]
                    if ch == '(':
                        inner_paren_count += 1
                    elif ch == ')':
                        inner_paren_count -= 1
                    elif ch == ',' and inner_paren_count == 0:
                        inner_parts.append(inner_part.strip())
                        inner_part = ""
                        continue
                    inner_part += ch
                
                if inner_part:
                    inner_parts.append(inner_part.strip())
                
                result.append(inner_parts)
                current_clause = ""
        elif c == ',' and paren_count == 0:
            if current_clause:
                result.append([current_clause.strip()])
                current_clause = ""
        else:
            current_clause += c
        
        i += 1
    
    if current_clause:
        result.append([current_clause.strip()])
    
    return result

def replace_substrings(input_string, replacement_dict):     #根据字典替换字符串中的子字符串
    sorted_keys = sorted(replacement_dict.keys(), key=len, reverse=True)
    
    pattern = '|'.join(map(re.escape, sorted_keys))
    
    def replacer(match):
        return replacement_dict[match.group(0)]
    
    return re.sub(pattern, replacer, input_string)

def dict_to_str(d: dict):   #将字典转换为字符串表示
    result_str = ""
    for key, value in d.items():
        result_str += f"({key}={value})"
    return result_str

if __name__ == '__main__':
    kb1= [['GradStudent(sue)',], ['~GradStudent(x)', 'Student(x)'], ['~Student(x)', 'HardWorker(x)'],
       ['~HardWorker(sue)',]]
    kb1=convert_list_to_FOL(kb1)

    kb2 = [
    ['A(tony)'], 
    ['A(mike)'], 
    ['A(john)'], 
    ['L(tony,rain)'], 
    ['L(tony,snow)'], 
    ['~A(x)', 'S(x)', 'C(x)'],
    ['~C(y)', '~L(y,rain)'], 
    ['L(z,snow)', '~S(z)'], 
    ['~L(tony,u)', '~L(mike,u)'], 
    ['L(tony,v)', 'L(mike,v)'],
    ['~A(w)', '~C(w)', 'S(w)']
    ]
    kb2 = convert_list_to_FOL(kb2)

    kb3 =[['On(tony,mike)'],['On(mike,john)'],['Green(tony)'],['~Green(john)'],['~On(xx,yy)','~Green(xx)','Green(yy)']]
    kb3 = convert_list_to_FOL(kb3)

    resolution1 = resolution_algorithm(kb1)
    print("---------------------------例题：------------------------------")
    if resolution1 == None:
        print("No solution")
    else:
        resolution1 = remove_unused_steps(resolution1, len(resolution1) - 1)
        for i in range(len(resolution1)):
            print(f"{i+1}:{resolution1[i]}")

    resolution2 = resolution_algorithm(kb2)
    print("---------------------------作业1：------------------------------")
    if resolution2 == None:
        print("No solution")
    else:
        resolution2 = remove_unused_steps(resolution2, len(resolution2) - 1)
        for i in range(len(resolution2)):
            print(f"{i+1}:{resolution2[i]}")

    resolution3 = resolution_algorithm(kb3)
    print("----------------------------作业2：-----------------------------")
    if resolution3 == None:
        print("No solution")
    else:
        resolution3 = remove_unused_steps(resolution3, len(resolution3) - 1)
        for i in range(len(resolution3)):
            print(f"{i+1}:{resolution3[i]}")