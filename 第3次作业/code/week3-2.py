import re

def unify_term(term1, term2, substitution):
    """尝试统一两个项 term1 和 term2，返回新的替换字典"""
    if term1 == term2:
        return substitution     #如果两项相同，直接替换
    elif is_variable(term1):    #如果 term1 是变量
        return unify_variable(term1, term2, substitution)
    elif is_variable(term2):    #如果 term2 是变量
        return unify_variable(term2, term1, substitution)
    elif term1.startswith('(') and term2.startswith('('):
        return unify_function(term1, term2, substitution)   #如果两项都是函数
    else:
        return None #无法统一


def unify_variable(var, term, substitution):
    """统一变量与其他项"""
    if var in substitution: #如果变量已经有替换
        return unify_term(substitution[var], term, substitution)
    if term == var:
        return substitution #如果变量已经是该项，直接返回
    if occurs_check(var, term):
        return None
    substitution[var] = term    #将变量替换为项
    return substitution


def occurs_check(var, term):
    """检查变量是否出现在项中"""
    if is_variable(term):
        return term == var  #如果项是变量，直接比较
    elif term.startswith('('):  #如果项是函数
        return any(occurs_check(var, subterm) for subterm in split_function(term))
    return False


def unify_function(func1, func2, substitution):
    """统一函数"""
    if func1[0] != func2[0]:
        return None #函数名不同，无法统一
    args1 = split_function(func1)
    args2 = split_function(func2)
    for arg1, arg2 in zip(args1, args2):
        substitution = unify_term(arg1, arg2, substitution) #逐个统一参数
        if substitution is None:
            return None
    return substitution

def split_function(func):
    """将函数表达式分割为参数列表"""
    assert func[0] == '(' and func[-1] == ')', "Function should start with '(' and end with ')'"
    body = func[1:-1]   #去掉括号
    level = 0
    parts = []  #存储分割之后的参数
    current = ''
    for char in body:
        if char == ',' and level == 0:
            parts.append(current)
            current = ''
        else:
            current += char
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
    parts.append(current)  #直接添加最后一个参数
    return parts

def is_variable(term):
    """检查是否是变量"""
    return term.islower()

def MGU(formula1, formula2):
    """最一般合一函数"""
    substitution = {}   #初始化替换字典
    pred1, args1 = parse_formula(formula1)
    pred2, args2 = parse_formula(formula2)
    if pred1 != pred2:
        return {}   #如果谓词不同，无法合一
    for arg1, arg2 in zip(args1, args2):
        substitution = unify_term(arg1, arg2, substitution) #统一每个参数
        if substitution is None:
            return {}   #如果某个参数无法合一，直接返回
    return substitution

def parse_formula(formula):
    """解析公式，将谓词和参数提取出来"""
    match = re.match(r'([A-Za-z]+)\((.*)\)', formula)
    if match:
        pred = match.group(1)
        args = match.group(2).split(',')
        return pred, args
    return None, None

# 示例测试
print(MGU('P(xx,a)', 'P(b,yy)'))  # 输出 {'xx': 'b', 'yy': 'a'}
print(MGU('P(a,xx,f(g(yy)))', 'P(2zz,f(zz),f(uu))'))  # 输出 {'zz': 'a', 'xx': 'f(a)', 'uu': 'g(yy)'}

