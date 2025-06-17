
import pdb
import os, re
import random
import itertools
from itertools import product, chain
from collections import defaultdict, namedtuple
import re
import sqlite3
from typing import Tuple, Any, List, Set, Literal, Iterator
import time
from contextlib import contextmanager


WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
Token = namedtuple('Token', ['ttype', 'value'])
VALUE_NUM_SYMBOL = 'VALUERARE'
TIMEOUT = 15


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def get_scores(count, pred_total, label_total):
    if pred_total != label_total:
        return 0, 0, 0
    elif count == pred_total:
        return 1, 1, 1
    return 0, 0, 0


def eval_sel(pred, label):
    pred_sel = pred['select'][1]
    label_sel = label['select'][1]
    label_wo_agg = [unit[1] for unit in label_sel]
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_where(pred, label):
    pred_conds = [unit for unit in pred['where'][::2]]
    label_conds = [unit for unit in label['where'][::2]]
    label_wo_agg = [unit[2] for unit in label_conds]
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_group(pred, label):
    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
    label_cols = [label.split(".")[1] if "." in label else label for label in label_cols]
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    return label_total, pred_total, cnt


def eval_having(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['groupBy']) > 0:
        pred_total = 1
    if len(label['groupBy']) > 0:
        label_total = 1

    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    if pred_total == label_total == 1 \
            and pred_cols == label_cols \
            and pred['having'] == label['having']:
        cnt = 1

    return label_total, pred_total, cnt


def eval_order(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['orderBy']) > 0:
        pred_total = 1
    if len(label['orderBy']) > 0:
        label_total = 1
    if len(label['orderBy']) > 0 and pred['orderBy'] == label['orderBy'] and \
            ((pred['limit'] is None and label['limit'] is None) or (pred['limit'] is not None and label['limit'] is not None)):
        cnt = 1
    return label_total, pred_total, cnt


def eval_and_or(pred, label):
    pred_ao = pred['where'][1::2]
    label_ao = label['where'][1::2]
    pred_ao = set(pred_ao)
    label_ao = set(label_ao)

    if pred_ao == label_ao:
        return 1, 1, 1
    return len(pred_ao), len(label_ao), 0


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def eval_nested(pred, label):
    label_total = 0
    pred_total = 0
    cnt = 0
    if pred is not None:
        pred_total += 1
    if label is not None:
        label_total += 1
    if pred is not None and label is not None:
        cnt += Executor().eval_exact_match(pred, label)
    return label_total, pred_total, cnt


def eval_IUEN(pred, label):
    lt1, pt1, cnt1 = eval_nested(pred['intersect'], label['intersect'])
    lt2, pt2, cnt2 = eval_nested(pred['except'], label['except'])
    lt3, pt3, cnt3 = eval_nested(pred['union'], label['union'])
    label_total = lt1 + lt2 + lt3
    pred_total = pt1 + pt2 + pt3
    cnt = cnt1 + cnt2 + cnt3
    return label_total, pred_total, cnt


def get_keywords(sql):
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')

    # or keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('in')]) > 0:
        res.add('in')

    # like keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')]) > 0:
        res.add('like')

    return res


def eval_keywords(pred, label):
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, pred_total, cnt


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                               [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count


def replace_cur_year(query: str) -> str:
    return re.sub(
        "YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE
    )

# get the database cursor for a sqlite database path


def get_cursor_from_path(sqlite_path: str):

    try:
        if not os.path.exists(sqlite_path):
            raise ValueError('sqlite path not exists.')
        connection = sqlite3.connect(sqlite_path)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor


def get_connection_from_sql(sql_script: str):
    try:
        with open(sql_script, "r") as fd:
            sql = fd.read()
        connection = sqlite3.connect(":memory:")
        connection.executescript(sql)
    except Exception as e:
        print(sql_script)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    return connection


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))

# unorder each row in the table
# [result_1 and result_2 has the same bag of unordered row]
# is a necessary condition of
# [result_1 and result_2 are equivalent in denotation]


def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])

# return whether two bag of relations are equivalent


def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


# check whether two denotations are correct
def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    if len(result1) == 0 and len(result2) == 0:
        return True

    # if length is not the same, then they are definitely different bag of rows
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])

    # if the results do not have the same number of columns, they are different
    if len(result2[0]) != num_cols:
        return False

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if not quick_rej(result1, result2, order_matters):
        return False

    # the rest of the problem is in fact more complicated than one might think
    # we want to find a permutation of column order and a permutation of row order,
    # s.t. result_1 is the same as result_2
    # we return true if we can find such column & row permutations
    # and false if we cannot
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
    # we decrease the size of the column permutation space by the function get_constraint_permutation
    # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            # in fact the first condition must hold if the second condition holds
            # but the first is way more efficient implementation-wise
            # and we use it to quickly reject impossible candidates
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False


def program_extract(text: str,
                    program: str = "python",
                    mode: Literal["first", "last", "all"] = "all") -> str:

    program_pattern = rf"```{program}[ \t]*[\r\n]+(.*?)[\r\n]+[ \t]*```"
    program_re = re.compile(program_pattern, re.DOTALL | re.IGNORECASE)
    matches = program_re.findall(text)
    if matches:
        if mode == "first":
            return matches[0].strip()
        elif mode == "last":
            return matches[-1].strip()
        elif mode == "all":
            return "\n\n".join(matches)
    else:
        print(f'INVALID:{text}')
        return "INVALID!"

# context manager for select timeout


@contextmanager
def sqlite_timelimit(conn, ms):
    deadline = time.perf_counter() + (ms / 1000)
    # n is the number of SQLite virtual machine instructions that will be
    # executed between each check. It takes about 0.08ms to execute 1000.
    # https://github.com/simonw/datasette/issues/1679
    n = 1000
    if ms <= 20:
        # This mainly happens while executing our test suite
        n = 1

    def handler():
        if time.perf_counter() >= deadline:
            # Returning 1 terminates the query with an error
            return 1

    conn.set_progress_handler(handler, n)
    try:
        yield
    finally:
        conn.set_progress_handler(None, n)


# extract the non-value tokens and the set of values
# from a sql query
def extract_query_values(sql: str) -> Tuple[List[str], Set[str]]:
    import sqlparse

    # strip_query, reformat_query and replace values
    # were implemented by Yu Tao for processing CoSQL
    def strip_query(query: str) -> Tuple[List[str], List[str]]:
        query_keywords, all_values = [], []
        # then replace all stuff enclosed by "" with a numerical value to get it marked as {VALUE}
        # Tao's implementation is commented out here.
        """
        str_1 = re.findall("\"[^\"]*\"", query)
        str_2 = re.findall("\'[^\']*\'", query)
        values = str_1 + str_2
        """

        toks = sqlparse.parse(query)[0].flatten()
        values = [t.value for t in toks if
                  t.ttype == sqlparse.tokens.Literal.String.Single or t.ttype == sqlparse.tokens.Literal.String.Symbol]

        for val in values:
            all_values.append(val)
            query = query.replace(val.strip(), VALUE_NUM_SYMBOL)

        query_tokenized = query.split()
        float_nums = re.findall("[-+]?\d*\.\d+", query)
        all_values += [qt for qt in query_tokenized if qt in float_nums]
        query_tokenized = [VALUE_NUM_SYMBOL if qt in float_nums else qt for qt in query_tokenized]

        query = " ".join(query_tokenized)
        int_nums = [i.strip() for i in re.findall("[^tT]\d+", query)]

        all_values += [qt for qt in query_tokenized if qt in int_nums]
        query_tokenized = [VALUE_NUM_SYMBOL if qt in int_nums else qt for qt in query_tokenized]
        # print int_nums, query, query_tokenized

        for tok in query_tokenized:
            if "." in tok:
                table = re.findall("[Tt]\d+\.", tok)
                if len(table) > 0:
                    to = tok.replace(".", " . ").split()
                    to = [t.lower() for t in to if len(t) > 0]
                    query_keywords.extend(to)
                else:
                    query_keywords.append(tok.lower())

            elif len(tok) > 0:
                query_keywords.append(tok.lower())
        return query_keywords, all_values
    def tokenize(query: str) -> List[Token]:
        tokens = list([Token(t.ttype, t.value) for t in sqlparse.parse(query)[0].flatten()])
        return tokens
    def reformat_query(query: str) -> str:
        query = query.strip().replace(";", "").replace("\t", "")
        query = ' '.join([t.value for t in tokenize(query) if t.ttype != sqlparse.tokens.Whitespace])
        t_stars = ["t1.*", "t2.*", "t3.*", "T1.*", "T2.*", "T3.*"]
        for ts in t_stars:
            query = query.replace(ts, "*")
        return query

    def replace_values(sql: str) -> Tuple[List[str], Set[str]]:
        sql = sqlparse.format(sql, reindent=False, keyword_case='upper')
        # sql = re.sub(r"(<=|>=|!=|=|<|>|,)", r" \1 ", sql)
        sql = re.sub(r"(T\d+\.)\s", r"\1", sql)
        query_toks_no_value, values = strip_query(sql)
        return query_toks_no_value, set(values)

    reformated = reformat_query(query=sql)
    query_value_replaced, values = replace_values(reformated)
    return query_value_replaced, values


# given the gold query and the model prediction
# extract values from the gold, extract predicted sql with value slots
# return 1) number of possible ways to plug in gold values and 2) an iterator of predictions with value plugged in
def get_all_preds_for_execution(gold: str, pred: str) -> Tuple[int, Iterator[str]]:

    # plug in the values into query with value slots
    def plugin(query_value_replaced: List[str], values_in_order: List[str]) -> str:
        q_length = len(query_value_replaced)
        query_w_values = query_value_replaced[:]
        value_idx = [idx for idx in range(q_length) if query_value_replaced[idx] == VALUE_NUM_SYMBOL.lower()]
        assert len(value_idx) == len(values_in_order)

        for idx, value in zip(value_idx, values_in_order):
            query_w_values[idx] = value
        return ' '.join(query_w_values)

    # a generator generating all possible ways of
    # filling values into predicted query
    def plugin_all_permutations(query_value_replaced: List[str], values: Set[str]) -> Iterator[str]:
        num_slots = len([v for v in query_value_replaced if v == VALUE_NUM_SYMBOL.lower()])
        for values in itertools.product(*[list(values) for _ in range(num_slots)]):
            yield plugin(query_value_replaced, list(values))

    _, gold_values = extract_query_values(gold)
    pred_query_value_replaced, _ = extract_query_values(pred)
    num_slots = len([v for v in pred_query_value_replaced if v == VALUE_NUM_SYMBOL.lower()])
    num_alternatives = len(gold_values) ** num_slots
    return num_alternatives, plugin_all_permutations(pred_query_value_replaced, gold_values)


class Executor:
    """A simple evaluator"""

    def __init__(self):
        self.partial_scores = None
        self.db_con_pool = {}

    def get_db_connection(self, db_id):
        if db_id not in self.db_con_pool:
            #'file:myDB.sqlite?immutable=1'
            conn = sqlite3.connect(f"file:{db_id}?immutable=1", uri=True, check_same_thread=False)
            conn.execute('pragma mmap_size=16294967296;')
            conn.execute('pragma journal_mode=OFF;')
            conn.execute('pragma synchronous=OFF;')
            conn.execute('PRAGMA temp_store=2;')

            conn.text_factory = lambda b: b.decode(errors="ignore")

            self.db_con_pool[db_id] = conn
            return conn
        else:
            return self.db_con_pool[db_id]

    def close_db_connection(self, db_id):
        assert db_id in self.db_con_pool
        conn = self.db_con_pool.pop(db_id)
        conn.close()

    def exec_sql(self, db_file, sql_strs, plug_value=False, keep_distinct=False, do_post_process=True):

        def postprocess(query: str) -> str:
            # extract sql
            query = program_extract(query, program='sql', mode='last')
            query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
            return query

        def remove_distinct(s):
            import sqlparse
            toks = [t.value for t in list(sqlparse.parse(s)[0].flatten())]
            return ''.join([t for t in toks if t.lower() != 'distinct'])

        def exec_on_db(
            sqlite_path: str, query: str, process_id: str = "", timeout: int = TIMEOUT
        ) -> Tuple[str, Any]:

            query = replace_cur_year(query)
            if ".sqlite" in sqlite_path:
                conn = self.get_db_connection(sqlite_path)

            elif ".sql" in sqlite_path:
                # 这个地方我不知道什么用，所以connection pool 没有加入
                conn = get_connection_from_sql(sqlite_path)
            else:
                raise NotImplementedError(f"{sqlite_path} is not supported yet!")

            try:
                #print(query)
                with sqlite_timelimit(conn, timeout * 1000):
                    cursor = conn.cursor()
                   
                    cursor.execute(query)
                    result = cursor.fetchall()

                    res = ('result', result)

            except sqlite3.OperationalError as e:

                if f"{e}".strip() == 'interrupted':
                    result = "timeout"
                else:
                    result = f"error:{e}"
                res = ('error', result)
            except Exception as e:
                self.close_db_connection(sqlite_path)
                cursor = None
                res = ('exception', e)
            finally:
                if cursor is not None:
                    cursor.close()

            return res

        results = []
        for sql_str in sql_strs:
            if do_post_process:
                sql_str = postprocess(sql_str)
            if not keep_distinct:
                sql_str = remove_distinct(sql_str)

            # we decide whether two denotations are equivalent based on "bag semantics"
            # https://courses.cs.washington.edu/courses/cse444/10sp/lectures/lecture16.pdf
            # if there is order by in query, then we assume order of the rows matter
            # order by might also be used to find the max/min instead of sorting,
            # but in that case the result mostly only contains one row and hence order_matters does not make a difference
            # order_matters = 'order by' in g_str.lower()

            # if plug in value (i.e. we do not consider value prediction correctness)
            # if plug_value:
            #     _, preds = get_all_preds_for_execution(gold, sql_str)
            #     preds = chain([sql_str], preds)

            flag, p_denotation = exec_on_db(db_file, sql_str)
            results.append((flag, p_denotation))
        return results

    @staticmethod
    def result_equal(result1: List[Tuple], result2: List[Tuple], order_matters: bool = False):
        return result_eq(result1, result2, order_matters)

    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"

    def eval_exact_match(self, pred, label):
        partial_scores = self.eval_partial_match(pred, label)
        self.partial_scores = partial_scores

        for key, score in partial_scores.items():
            if score['f1'] != 1:
                return 0

        if len(label['from']['table_units']) > 0:
            label_tables = sorted(label['from']['table_units'])
            pred_tables = sorted(pred['from']['table_units'])
            return label_tables == pred_tables
        return 1

    def eval_partial_match(self, pred, label):
        res = {}

        label_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['select'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['select(no AGG)'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['where'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['where(no OP)'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_group(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group(no Having)'] = {'acc': acc, 'rec': rec, 'f1': f1,
                                   'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_having(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_order(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['order'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_and_or(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['and/or'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_IUEN(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['IUEN'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_keywords(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['keywords'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        return res




excecutor = Executor()
eq_func = Executor.result_equal

import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

def with_timeout(timeout_seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 设置信号处理
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # 重置闹钟
                signal.alarm(0)
            
            return result
        return wrapper
    return decorator


def score(solution_str,ground_truth, executor):
    query = solution_str
    trajs = ground_truth
    gold_sql = trajs['gold_sql']
    db_id = os.path.join('/cache/', trajs['db_id'])
    
    cmp_method = trajs.get('cmp_method', 'spider')
    gold_answer = excecutor.exec_sql(db_id, [gold_sql],
                                            keep_distinct=True,
                                            do_post_process=False)
    if gold_answer[0][0] != 'result':
        print(f"db_id:{db_id} gold_sql: {gold_sql} is not executable. result:{gold_answer}")
        message = ''
        flag = False
    
    exec_res = []
    res = excecutor.exec_sql(db_id, [query], keep_distinct=True)
    if res[0][0] != 'result':
        message = f"Execution Error: {str(res[0][1]).split('error:')[-1]}"
        flag = False 
    else:
        if cmp_method == "spider":
            order_matters = "orderby" in re.sub(r"\s+", gold_sql.lower(), "")
            flag = eq_func(res[0][-1], gold_answer[0][-1], order_matters)
            
        else:
            flag = set(res[0][-1]) == set(gold_answer[0][-1])
        message = ""

    return (1.0 if flag else 0.0), message

if __name__ == "__main__":
    # Example usage
    query = "```sql\nSELECT T2.`T-BIL`, T1.ID, T1.SEX, T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID ORDER BY T2.`T-BIL` DESC LIMIT 1\n```\n"
    trajs = {
        "db_id": "thrombosis_prediction/thrombosis_prediction.sqlite", 
        "gold_sql": "SELECT T2.`T-BIL`, T1.ID, T1.SEX, T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID ORDER BY T2.`T-IL` DESC LIMIT 1", 
        "cmp_method": "bird"}
    score_result = score(query, trajs)
    print("Score result:", score_result)