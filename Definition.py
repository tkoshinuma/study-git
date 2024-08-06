import requests
import os
import zipfile
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import glob
from datetime import datetime, timedelta
#コメント追加
# EDINET APIと通信
def get_submitted_summary(params1):
    url = "https://api.edinet-fsa.go.jp/api/v2" + '/documents.json'
    response = requests.get(url, params1)
     
    # responseが200でなければエラーを出力
    assert response.status_code==200
       
    return response.json()

# csv用にEDINET APIと通信
def get_csv_document(doc_id, params2):
    url = "https://api.edinet-fsa.go.jp/api/v2" + '/documents/' + doc_id
    response = requests.get(url, params2)
     
    return response

# csvをダウンロード
def download_document(doc_id, save_path, params2):
    doc = get_csv_document(doc_id, params2)
    if doc.status_code == 200:
        with open(save_path + doc_id + '.zip', 'wb') as f:
            for chunk in doc.iter_content(chunk_size=1024):
                f.write(chunk)

# 有報を全取得する(use)
def download_all_documents(start_date, end_date, save_path, doc_type_codes=['120']):
    # start_date, end_dateには期間をdatetime型で、save_pathにはファイル名を入れる    
    # doc_type_codesは、文書の種類で120が有価証券報告書、130が訂正有価証券報告書
    for i, date in enumerate(date_range(start_date, end_date)):
        date_str = str(date)[:10]
        params1 = {'date': date_str, 'type': 2, 'Subscription-Key': "6270bd3b5b124e61860adfaa878d0c36"}
        params2 = {'type': 5, 'Subscription-Key': "6270bd3b5b124e61860adfaa878d0c36"}
        doc_summary = get_submitted_summary(params1)
        df_doc_summary = pd.DataFrame(doc_summary['results'])
        df_meta = pd.DataFrame(doc_summary['metadata'])
         
        # 対象とする報告書のみ抽出
        if len(df_doc_summary) >= 1:
            df_doc_summary = df_doc_summary.loc[df_doc_summary['docTypeCode'].isin(doc_type_codes)]#doc_type_codesに入っているものだけを探す
            df_doc_summary = df_doc_summary[df_doc_summary['fundCode'].isnull()]
            # ここに銘柄コードを入れることで、単体でとれる
            # df_doc_summary.dropna(subset = ["secCode"])
            # df_doc_summary = df_doc_summary[df_doc_summary["secCode"] == "78160"]     
     
            # 一覧を保存
            if not os.path.exists(save_path + date_str):
                os.makedirs(save_path + date_str)
            df_doc_summary.to_csv(save_path + date_str + '/doc_summary.csv')       
            # time.sleep(1)  # 単体の時はsleepしないと引っかかる。
            
            # 書類を保存
            for _, doc in df_doc_summary.iterrows():
                # print(doc["docID"])
                download_document(doc['docID'], save_path + date_str + '/', params2)
                open_zip_file(doc['docID'], save_path + date_str + '/')

# Zipファイルを解凍
def open_zip_file(doc_id, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path + doc_id)
    try:    
        with zipfile.ZipFile(save_path + doc_id + '.zip') as zip_f:
            zip_f.extractall(save_path + doc_id)
    except Exception as e:
        print(e)

# 日付指定
def date_range(start_date, end_date):
    diff = (end_date - start_date).days + 1
    
    return (start_date + timedelta(i) for i in range(diff))

# EDINETタクソノミを引数にkey一覧の作成、一般が先頭に来るように設定しないと、数値チェックの時にバグる。
def all_key(input_book):
    df_key_list = pd.DataFrame()
    input_sheet_name = input_book.sheet_names
    for i in range(3, len(input_sheet_name)):
        input_sheet_df = input_book.parse(input_sheet_name[i])
        input_sheet_df.columns = input_sheet_df.iloc[0]
        input_sheet_df = input_sheet_df.drop(input_sheet_df.index[0])
        input_sheet_df["要素ID"] = input_sheet_df["名前空間プレフィックス"] + ":" + input_sheet_df["要素名"]
        input_sheet_df = input_sheet_df[["要素ID", "JASA"]]
        df_key_list = pd.concat([df_key_list, input_sheet_df], ignore_index = True)
    df_key_list = df_key_list.drop_duplicates(subset='要素ID', ignore_index = True)

    return df_key_list

# another_versionを作成する
def all_key1(input_book):
    df_key_list = pd.DataFrame()
    input_sheet_name = input_book.sheet_names
    for i in range(3, len(input_sheet_name)):
        input_sheet_df = input_book.parse(input_sheet_name[i])
        input_sheet_df.columns = input_sheet_df.iloc[0]
        input_sheet_df = input_sheet_df.drop(input_sheet_df.index[0])
        input_sheet_df["要素ID"] = input_sheet_df["要素名"]
        input_sheet_df = input_sheet_df[["要素ID", "JASA"]]
        df_key_list = pd.concat([df_key_list, input_sheet_df], ignore_index = True)
    df_key_list = df_key_list.drop_duplicates(subset='要素ID', ignore_index = True)

    return df_key_list


# EDINETタクソノミを引数に個別の作成
def indivisual_key(input_book, num):
    df_key_list = pd.DataFrame()
    input_sheet_name = input_book.sheet_names
    input_sheet_df = input_book.parse(input_sheet_name[num])
    input_sheet_df.columns = input_sheet_df.iloc[0]
    input_sheet_df = input_sheet_df.drop(input_sheet_df.index[0])
    input_sheet_df["要素ID"] = input_sheet_df["名前空間プレフィックス"] + ":" + input_sheet_df["要素名"]
    df_key_list = input_sheet_df[["要素ID", "JASA"]]

    return df_key_list

def multi_key(input_book, num_list):
    df_key_list = pd.DataFrame()
    input_sheet_name = input_book.sheet_names
    
    for num in num_list:
        input_sheet_df = input_book.parse(input_sheet_name[num])
        input_sheet_df.columns = input_sheet_df.iloc[0]
        input_sheet_df = input_sheet_df.drop(input_sheet_df.index[0]).reset_index(drop=True)
        input_sheet_df["要素ID"] = input_sheet_df["名前空間プレフィックス"] + ":" + input_sheet_df["要素名"]
        df_key_list_temp = input_sheet_df[["要素ID", "JASA"]]
        df_key_list = pd.concat([df_key_list, df_key_list_temp], ignore_index=True)
    df_key_list = df_key_list.drop_duplicates(subset='要素ID', ignore_index=True)

    return df_key_list

# ある期間での全企業の有報をDataFrame型でconcatして、ダウンロードはせず取得(use)
def extract_document(start_date, end_date, save_path, doc_type_codes=['120']):
    # start_date, end_dateには期間をdatetime型で、save_pathにはファイル名を入れる
    # doc_type_codesは、文書の種類で120が有価証券報告書、130が訂正有価証券報告書

    for i, date in enumerate(date_range(start_date, end_date)):
        date_str = str(date)[:10]
        params1 = {'date': date_str, 'type': 2, 'Subscription-Key': "6270bd3b5b124e61860adfaa878d0c36"}
        params2 = {'type': 5, 'Subscription-Key': "6270bd3b5b124e61860adfaa878d0c36"}
        doc_summary = get_submitted_summary(params1)
        df_doc_summary = pd.DataFrame(doc_summary['results'])
        df_meta = pd.DataFrame(doc_summary['metadata'])
        
        # 対象とする報告書のみ抽出
        if len(df_doc_summary) >= 1:
            df_doc_summary = df_doc_summary.loc[df_doc_summary['docTypeCode'].isin(doc_type_codes)]#doc_type_codesに入っているものだけを探す
            df_doc_summary = df_doc_summary[df_doc_summary['fundCode'].isnull()]
            # df_doc_summary.dropna(subset = ["secCode"])
            # df_doc_summary = df_doc_summary[df_doc_summary["secCode"] == "78160"] 
            # ここに銘柄コードを入れることで、単体でとれる
        
        if i == 0:
            df_doc_summary_all = df_doc_summary.copy()
        else:
            df_doc_summary_all = pd.concat([df_doc_summary_all, df_doc_summary])

    return df_doc_summary_all

# 各企業の有報のdoc_idとdateを取ってくる
def search_doc_ID(df, seccode):
    try:
        doc_id = df.query(f'secCode=="{seccode}0"').docID.values[0]
        #print(doc_id)
        date = df.query(f'secCode=="{seccode}0"').submitDateTime.values[0]
        #print(date)
        
        return doc_id, date
    except Exception as e:
        print(e)
        
        return 0, 0

# csvファイルを解凍
def unfreeze_csv(save_path, doc_id, date):
    try:
        csvfile = save_path + date[:10] + '/' + doc_id + '/XBRL_TO_CSV/jpcrp*.csv'
        csvfile = glob.glob(csvfile)[0]
        
        return csvfile
    except Exception as e:
        print(e)
        
        return 0

# 対象企業の財務情報を一括取得(use)
def get_number(save_path, df_doc_summary_all, seccode_list): 
    # df_doc_summary_allにはextract_documentで得たものを、seccode_listにはlist型で

    for i, code in enumerate(seccode_list):
        doc_id, date = search_doc_ID(df_doc_summary_all, code)  
        time.sleep(1)
        # csvを取得
        if doc_id != 0:
            csvfile = unfreeze_csv(save_path, doc_id, date)
            if csvfile != 0:        
                df = pd.read_csv(csvfile, encoding = "utf-16", on_bad_lines="skip", sep = "\t")
                df = df[["要素ID", "コンテキストID", "値"]]

                # 数値の取得
                df_duration = df[df['コンテキストID'].isin(['CurrentYearDuration'])]
                df_instant = df[df['コンテキストID'].isin(['CurrentYearInstant'])]
                df_date = df[df['要素ID'].isin(['jpdei_cor:CurrentPeriodEndDateDEI'])]
                df_other1 = df[df['要素ID'].str.contains('jpcrp030000-')]
                df_other2 = df[df["要素ID"].str.contains('jppfs_cor:')]

                df_temp = pd.concat([df_duration, df_instant])
                df_temp2 = pd.concat([df_temp, df_date])
                df_temp2 = df_temp2.drop_duplicates(subset='要素ID')
                df_other1 = df_other1.drop_duplicates(subset="要素ID")
                df_other2 = df_other2.drop_duplicates(subset="要素ID")
                df_other = pd.concat([df_other1, df_other2])

                if i == 0:
                    df_merge= df_temp2.copy()
                    df_merge = df_merge.rename(columns = {"値":code})
                    df_merge = df_merge.drop("コンテキストID", axis = 1)
                    filtered_df = df_other.copy()
                    filtered_df = filtered_df.rename(columns = {"値":code})
                    filtered_df = filtered_df.drop("コンテキストID", axis = 1)
                    
                else:
                    df_merge = pd.merge(df_merge, df_temp2, on = "要素ID", how = "outer")
                    df_merge = df_merge.rename(columns = {"値":code})
                    df_merge = df_merge.drop("コンテキストID", axis = 1)
                    filtered_df = pd.merge(filtered_df, df_other, on = "要素ID", how = "outer")
                    filtered_df = filtered_df.rename(columns = {"値":code})
                    filtered_df = filtered_df.drop("コンテキストID", axis = 1)
                    
            else:
                print(code)  # 失敗した銘柄とその理由を表示

                if 'df_merge' in locals():
                    if not df_merge.empty:
                        df_merge[code] = np.nan
                    else:
                        df_merge = pd.DataFrame({code: [np.nan]})
                elif 'filtered_df' in locals():
                    if not filtered_df.empty:
                        filtered_df[code] = np.nan
                    else:
                        filtered_df = pd.DataFrame({code: [np.nan]})

        else:
            print(code)  # 失敗した銘柄とその理由を表示

            # df_mergeに新しい列を追加し、全てのセルにNaNを設定
            if 'df_merge' in locals():
                if not df_merge.empty:
                    df_merge[code] = np.nan
                else:
                    df_merge = pd.DataFrame({code: [np.nan]})
            elif 'filtered_df' in locals():
                if not filtered_df.empty:
                    filtered_df[code] = np.nan
                else:
                    filtered_df = pd.DataFrame({code: [np.nan]})

    df_info = df_merge.replace('－', 0.1)
    filtered_df = filtered_df.replace('－', 0.1)
    filtered_df.set_index("要素ID", inplace=True)
                
    
    return df_info, filtered_df

# 対象企業の財務情報を一括取得(use)
def get_number_NonConsolidated(save_path, df_doc_summary_all, seccode_list): 
    # df_doc_summary_allにはextract_documentで得たものを、seccode_listにはlist型で

    for i, code in enumerate(seccode_list):
        doc_id, date = search_doc_ID(df_doc_summary_all, code)  
        time.sleep(1)
        # csvを取得
        if doc_id != 0:
            csvfile = unfreeze_csv(save_path, doc_id, date)
            if csvfile != 0:        
                df = pd.read_csv(csvfile, encoding = "utf-16", on_bad_lines="skip", sep = "\t")
                df = df[["要素ID", "コンテキストID", "値"]]

                # 数値の取得
                df_duration = df[df['コンテキストID'].isin(['CurrentYearDuration_NonConsolidatedMember'])]
                df_instant = df[df['コンテキストID'].isin(['CurrentYearInstant_NonConsolidatedMember'])]
                df_date = df[df['要素ID'].isin(['jpdei_cor:CurrentPeriodEndDateDEI'])]
                df_other1 = df[df['要素ID'].str.contains('jpcrp030000-')]
                df_other2 = df[df["要素ID"].str.contains('jppfs_cor:')]

                df_temp = pd.concat([df_duration, df_instant])
                df_temp2 = pd.concat([df_temp, df_date])
                df_temp2 = df_temp2.drop_duplicates(subset='要素ID')
                df_other1 = df_other1.drop_duplicates(subset="要素ID")
                df_other2 = df_other2.drop_duplicates(subset="要素ID")
                df_other = pd.concat([df_other1, df_other2])

                if i == 0:
                    df_merge= df_temp2.copy()
                    df_merge = df_merge.rename(columns = {"値":code})
                    df_merge = df_merge.drop("コンテキストID", axis = 1)
                    filtered_df = df_other.copy()
                    filtered_df = filtered_df.rename(columns = {"値":code})
                    filtered_df = filtered_df.drop("コンテキストID", axis = 1)
                    
                else:
                    df_merge = pd.merge(df_merge, df_temp2, on = "要素ID", how = "outer")
                    df_merge = df_merge.rename(columns = {"値":code})
                    df_merge = df_merge.drop("コンテキストID", axis = 1)
                    filtered_df = pd.merge(filtered_df, df_other, on = "要素ID", how = "outer")
                    filtered_df = filtered_df.rename(columns = {"値":code})
                    filtered_df = filtered_df.drop("コンテキストID", axis = 1)
                    
            else:
                print(code)  # 失敗した銘柄とその理由を表示

                if 'df_merge' in locals():
                    if not df_merge.empty:
                        df_merge[code] = np.nan
                    else:
                        df_merge = pd.DataFrame({code: [np.nan]})
                elif 'filtered_df' in locals():
                    if not filtered_df.empty:
                        filtered_df[code] = np.nan
                    else:
                        filtered_df = pd.DataFrame({code: [np.nan]})

        else:
            print(code)  # 失敗した銘柄とその理由を表示

            # df_mergeに新しい列を追加し、全てのセルにNaNを設定
            if 'df_merge' in locals():
                if not df_merge.empty:
                    df_merge[code] = np.nan
                else:
                    df_merge = pd.DataFrame({code: [np.nan]})
            elif 'filtered_df' in locals():
                if not filtered_df.empty:
                    filtered_df[code] = np.nan
                else:
                    filtered_df = pd.DataFrame({code: [np.nan]})

    df_info = df_merge.replace('－', 0.1)
    filtered_df = filtered_df.replace('－', 0.1)
    filtered_df.set_index("要素ID", inplace=True)
                
    
    return df_info, filtered_df



# JASAフォーマットに落とし込む(use)
def JASA(df_info, df_key_list):
    # keyベースの数値データと、key一覧を引数に代入
    
    #df_info.columns = df_info.iloc[0]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "right")
    join_df = join_df.dropna(subset = "JASA") #ここを消した場合、別記事業codeが入って、error(float + str)が出るはず。ここが問題か？
    join_df = join_df.fillna(0)
    join_df = join_df.drop("要素ID", axis = 1)
    df_jasa = join_df.groupby(["JASA"]).sum()
    
    return df_jasa

# JASAで割り振られていない項目を％で確認(use)
def check_not_JASA(df_info, df_key_list, jasa_code, other_jasa_code):
    # keyベースの数値データと、key一覧を引数に代入、jasa_codeには知りたい割り振られていない項目、numには対応する合計値を代入
    #df_info.columns = df_info.iloc[0]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "right")
    #join_df = join_df.dropna(subset = "JASA") 
    join_df = join_df.fillna(0)
    df_check = join_df[join_df["JASA"].isin(other_jasa_code)]
    df_num = join_df[join_df["JASA"] == jasa_code]
    #df_check.iloc[:, 2:] = df_check.iloc[:, 2:].div(df_num) * 100

    #for i in range(2, len(df_check.index)):
        #df_check.iloc[:, i] = df_check.iloc[:, i].div

    #for i in range(2, len(df_num.columns)):
        #df_check.iloc[:, i] = np.where(df_num.iloc[0, i] == 0, 0, df_check.iloc[:, i] / df_num.iloc[0, i] * 100) 

    for i in range(2, len(df_num.columns)):
        df_check.iloc[:, i] = df_check.iloc[:, i].div(df_num.iloc[0, i], fill_value=0) * 100
    
    return df_check

def JASA_ca(df_info, df_key_list, filtered_df):
    jasa_ca_code = ["Cash and Short Term Investments", 
                "Notes Receivable/Accounts Receivable",
            "Notes Receivable/Accounts Receivable 1",
            "Notes Receivable/Accounts Receivable 2",
            "Notes Receivable/Accounts Receivable 3",
            "Notes Receivable/Accounts Receivable 4",
            "Notes Receivable/Accounts Receivable 5",
            "Notes Receivable/Accounts Receivable 6",
            "Notes Receivable/Accounts Receivable 7",
            "Electronically Recorded Monetary Claims",
            "Total Inventory",
            "Inventory1", 
            "Inventory2",
            "Inventory3",
            "Inventory4",
            "Inventory",
            "Prepaid Expenses",
            "Trading Securities",
            "Total Current Assets"]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
    #join_df = join_df.dropna(subset = "JASA") 
    join_df = join_df.fillna(0.1)
    df_jasa_ca = join_df[join_df["JASA"].isin(jasa_ca_code)]
    df_jasa_ca = df_jasa_ca.drop("要素ID", axis = 1)
    df_jasa_ca.iloc[:, 1:] = df_jasa_ca.iloc[:, 1:].astype(np.float64)
    df_jasa_ca = df_jasa_ca.groupby(["JASA"]).sum()

    inventory_values = []
    for column_name, item in df_jasa_ca.items():
        if item["Inventory1"] > 1:
            if item["Inventory3"] > 1:
                temp = item["Inventory1"] + item["Inventory3"] + item["Inventory"]
            else:
                temp = item["Inventory1"] + item["Inventory4"] + item["Inventory"]
        else:
            if item["Inventory3"] > 1:
                temp = item["Inventory2"] + item["Inventory3"] + item["Inventory"]
            else:
                temp = item["Inventory2"] + item["Inventory4"] + item["Inventory"]
        inventory_values.append(temp)

    df_jasa_ca.loc["Inventory True"] = inventory_values 

    #数値が大きい方を取ってきている。
    total_inventory_values = []
    for column_name, item in df_jasa_ca.items():
        if item["Total Inventory"] > item["Inventory True"]: 
            temp = item["Total Inventory"]
        else:
            temp = item["Inventory True"]
        total_inventory_values.append(temp)

    df_jasa_ca.loc["Total Inventory True"] = total_inventory_values

    receivable_values = []
    for column_name, item in df_jasa_ca.items():
        if item["Notes Receivable/Accounts Receivable 1"] > 1:
            temp = item["Notes Receivable/Accounts Receivable 1"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
        elif item["Notes Receivable/Accounts Receivable 2"] > 1:
            temp = item["Notes Receivable/Accounts Receivable 2"] + item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
        elif item["Notes Receivable/Accounts Receivable 3"] > 1:
            temp = item["Notes Receivable/Accounts Receivable 3"] + item["Notes Receivable/Accounts Receivable 4"] + item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
        elif item["Notes Receivable/Accounts Receivable 6"] > 1:
            temp = item["Notes Receivable/Accounts Receivable 6"] + item["Notes Receivable/Accounts Receivable 4"] + item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable"]
        elif item["Notes Receivable/Accounts Receivable 4"] > 1:
            temp = item["Notes Receivable/Accounts Receivable 4"] + item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
        elif item["Notes Receivable/Accounts Receivable 5"] > 1:
            temp = item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
        else:
            temp = item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
        receivable_values.append(temp)

    df_jasa_ca.loc["Notes Receivable/Accounts Receivable True"] = receivable_values

    # receivable_ca = []
    # for column_name, item in df_jasa_ca.items():
    #     temp = item["Notes Receivable/Accounts Receivable True"]
    #     if column_name == str(6532):
    #         temp += float(filtered_df.loc["jpcrp030000-asr_E32549-000:AccountsReceivableTradeAndContractAssetsCA", str(column_name)])
    #     receivable_ca.append(temp)
    
    # df_jasa_ca.loc["Notes Receivable/Accounts Receivable True"] = receivable_ca

    other_ca = []
    for column_name, item in df_jasa_ca.items():
        temp = item["Total Current Assets"] - item["Cash and Short Term Investments"] - item["Notes Receivable/Accounts Receivable True"] - item["Total Inventory True"] - item["Prepaid Expenses"] - item["Trading Securities"]
        other_ca.append(temp)

    
    df_jasa_ca.loc["Other Current Assets"] = other_ca

    df_jasa_ca = df_jasa_ca.reindex(index=["Total Current Assets", 
                                        "Cash and Short Term Investments",
                                        "Notes Receivable/Accounts Receivable True",
                                        "Total Inventory True",
                                        "Prepaid Expenses",
                                        "Trading Securities",
                                        "Other Current Assets",
                                        "Electronically Recorded Monetary Claims",
                                        "Notes Receivable/Accounts Receivable 1",
                                        "Notes Receivable/Accounts Receivable 2",
                                        "Notes Receivable/Accounts Receivable 3",
                                        "Notes Receivable/Accounts Receivable 4",
                                        "Notes Receivable/Accounts Receivable 5",
                                        "Notes Receivable/Accounts Receivable 6",
                                        "Notes Receivable/Accounts Receivable 7",
                                        "Notes Receivable/Accounts Receivable"])

    return df_jasa_ca

# 固定資産(Non-Current Assets)
def JASA_nca(df_info, df_key_list, filtered_df):
    jasa_nca_code = ["Total Non-Current Assets",
                    "Property/Plant/Equipment, Total - Net",
                    "Intangibles, Net",
                    "Goodwill, Net",
                    "Investments and Other Assets",
                    "Long Term Investments",
                    "Leasehold and Guarantee Deposits 1",
                    "Leasehold and Guarantee Deposits 2",
                    "Long Term Deposits",
                    "Deferred Tax Assets",
                    "Total Assets"]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
    #join_df = join_df.dropna(subset = "JASA") #ここを消した場合、別記事業codeが入って、error(float + str)が出るはず。でもここが問題か？
    join_df = join_df.fillna(0.1)
    df_jasa_nca = join_df[join_df["JASA"].isin(jasa_nca_code)]
    df_jasa_nca = df_jasa_nca.drop("要素ID", axis = 1)
    df_jasa_nca.iloc[:, 1:] = df_jasa_nca.iloc[:, 1:].astype(np.float64)
    df_jasa_nca = df_jasa_nca.groupby(["JASA"]).sum()

    lease_values = []
    for column_name, item in df_jasa_nca.items():
        if item["Leasehold and Guarantee Deposits 1"] > 1:
            temp = item["Leasehold and Guarantee Deposits 1"]
        else:
            temp = item["Leasehold and Guarantee Deposits 2"]
        lease_values.append(temp)

    df_jasa_nca.loc["Leasehold and Guarantee Deposits True"] = lease_values

    intangibles_other_values = []
    for column_name, item in df_jasa_nca.items():
        temp = item["Intangibles, Net"] - item["Goodwill, Net"]
        intangibles_other_values.append(temp)

    df_jasa_nca.loc["Other Intangibles, Net"] = intangibles_other_values

    other_ncl = []
    for column_name, item in df_jasa_nca.items():
        temp = item["Total Non-Current Assets"] - item["Property/Plant/Equipment, Total - Net"] - item["Goodwill, Net"] - item["Other Intangibles, Net"] - item["Long Term Investments"] - item["Leasehold and Guarantee Deposits True"] - item["Deferred Tax Assets"]
        other_ncl.append(temp)
    
    df_jasa_nca.loc["Other Non-Current Assets"] = other_ncl

    df_jasa_nca = df_jasa_nca.reindex(index=["Total Non-Current Assets",
                                            "Property/Plant/Equipment, Total - Net",
                                            "Goodwill, Net",
                                            "Other Intangibles, Net",
                                            "Long Term Investments",
                                            "Leasehold and Guarantee Deposits True",
                                            "Deferred Tax Assets",
                                            "Other Non-Current Assets",
                                            "Total Assets"])

    return df_jasa_nca

# 流動負債(Current Liabilities)
def JASA_cl(df_info, df_key_list, filtered_df):
    jasa_cl_code = ["Notes Payable/Accounts Payable",
                    "Notes Payable/Accounts Payable 1",
                "Notes Payable/Accounts Payable 2",
                "Notes Payable/Accounts Payable 3",
                "Notes Payable/Accounts Payable 4",
                "Notes Payable/Accounts Payable 5",
                "Notes Payable/Accounts Payable 6",
                "Notes Payable/Accounts Payable 7",
                "Electronically Recorded Monetary Debt",
                "Capital Leases",
                "Short Term Debt",
                "Current Port. of LT Debt",
                "Advances Received",
                "Total Current Liabilities",
                "Provision CL",
                "Provision CL, Total"]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
    join_df = join_df.fillna(0.1)
    df_jasa_cl = join_df[join_df["JASA"].isin(jasa_cl_code)]
    df_jasa_cl = df_jasa_cl.drop("要素ID", axis = 1)
    df_jasa_cl.iloc[:, 1:] = df_jasa_cl.iloc[:, 1:].astype(np.float64)
    df_jasa_cl = df_jasa_cl.groupby(["JASA"]).sum()

    notes_payable_values = []
    for column_name, item in df_jasa_cl.items():
        if item["Notes Payable/Accounts Payable 1"] > 1:
            if item["Notes Payable/Accounts Payable 6"] > 1:
                temp = item["Notes Payable/Accounts Payable 1"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
            else:
                temp = item["Notes Payable/Accounts Payable 1"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
        elif item["Notes Payable/Accounts Payable 2"]:
            if item["Notes Payable/Accounts Payable 6"] > 1:
                temp = item["Notes Payable/Accounts Payable 2"] + item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
            else:
                temp = item["Notes Payable/Accounts Payable 2"] + item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
        elif item["Notes Payable/Accounts Payable 5"] > 1:
            if item["Notes Payable/Accounts Payable 6"] > 1:
                temp = item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 5"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
            else:
                temp = item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 5"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
        elif item["Notes Payable/Accounts Payable 3"] > 1:
            if item["Notes Payable/Accounts Payable 6"] > 1:
                temp = item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
            elif item["Notes Payable/Accounts Payable 7"] > 1:
                temp = item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
        else:
            if item["Notes Payable/Accounts Payable 6"] > 1:
                temp = item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
            else:
                temp = item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
        notes_payable_values.append(temp)

    df_jasa_cl.loc["Notes Payable/Accounts Payable True"] = notes_payable_values  

    # advances_values = []
    # for column_name, item in df_jasa_cl.items():
    #     if item["Advances Received 2"] > 1:
    #         temp = item["Advances Received 2"]
    #     else:
    #         temp = item["Advances Received 1"]
    #     advances_values.append(temp)

    # df_jasa_cl.loc["Advances Received True"] = advances_values

    advances_values = []
    for column_name, item in df_jasa_cl.items():
        temp = item["Advances Received"]
        advances_values.append(temp)

    df_jasa_cl.loc["Advances Received True"] = advances_values

    short_values = []
    for column_name, item in df_jasa_cl.items():
        if item["Short Term Debt"] < item["Total Current Liabilities"] - (item["Notes Payable/Accounts Payable True"] + item["Advances Received True"] + item["Capital Leases"]):
            temp = item["Short Term Debt"]
        else:
            temp = 0
        short_values.append(temp)

    df_jasa_cl.loc["Short Term Debt"] = short_values

    long_values = []
    for column_name, item in df_jasa_cl.items():
        if item["Current Port. of LT Debt"] < item["Total Current Liabilities"] - (item["Notes Payable/Accounts Payable True"] + item["Advances Received True"] + item["Capital Leases"] + item["Short Term Debt"]):
            temp = item["Current Port. of LT Debt"]
        else:
            temp = 0
        long_values.append(temp)

    df_jasa_cl.loc["Current Port. of LT Debt"] = long_values

    provision_cl = []
    for column_name, item in df_jasa_cl.items():
        if item["Provision CL, Total"] > 1:
            temp = item["Provision CL, Total"]
        else:
            temp = item["Provision CL"]
        provision_cl.append(temp)
    
    df_jasa_cl.loc["Provision CL True"] = provision_cl

    other_cl = []
    for column_name, item in df_jasa_cl.items():
        temp = item["Total Current Liabilities"] - item["Notes Payable/Accounts Payable True"] - item["Advances Received True"] - item["Capital Leases"] - item["Short Term Debt"] - item["Current Port. of LT Debt"] - item["Provision CL True"]
        other_cl.append(temp)
    
    df_jasa_cl.loc["Other Current Liabilities"] = other_cl

    df_jasa_cl = df_jasa_cl.reindex(index=["Total Current Liabilities", 
                                        "Notes Payable/Accounts Payable True",
                                        "Advances Received True",
                                        "Capital Leases",
                                        "Short Term Debt",
                                        "Current Port. of LT Debt",
                                        "Other Current Liabilities",
                                        "Provision CL True",
                                        "Electronically Recorded Monetary Debt",
                                        "Notes Payable/Accounts Payable 1",
                                        "Notes Payable/Accounts Payable 2",
                                        "Notes Payable/Accounts Payable 3",
                                        "Notes Payable/Accounts Payable 4",
                                        "Notes Payable/Accounts Payable 5",
                                        "Notes Payable/Accounts Payable 6",
                                        "Notes Payable/Accounts Payable 7",
                                        "Notes Payable/Accounts Payable"]) 
    
    return df_jasa_cl

# 固定負債(Non-Current Liabilities)
def JASA_ncl(df_info, df_key_list, filtered_df):
    jasa_ncl_code = ["Total Non-Current Liabilities",
                    "Long Term Debt",
                    "Lease Liabilities",
                    "Retirement Benefit",
                    "Deferred Tax Liabilities",
                    "Asset Retirement Obligation",
                    "Bonds",
                    "Provision NCL, Total",
                    "Provision NCL"]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
    join_df = join_df.fillna(0.1)
    df_jasa_ncl = join_df[join_df["JASA"].isin(jasa_ncl_code)]
    df_jasa_ncl = df_jasa_ncl.drop("要素ID", axis = 1)
    df_jasa_ncl.iloc[:, 1:] = df_jasa_ncl.iloc[:, 1:].astype(np.float64)
    df_jasa_ncl = df_jasa_ncl.groupby(["JASA"]).sum()

    provision_ncl = []
    for column_name, item in df_jasa_ncl.items():
        if item["Provision NCL, Total"] > 1:
            temp = item["Provision NCL, Total"]
        else:
            temp = item["Provision NCL"]
        provision_ncl.append(temp)
    
    df_jasa_ncl.loc["Provision NCL True"] = provision_ncl


    other_ncl = []
    for column_name, item in df_jasa_ncl.items():
        temp = item["Total Non-Current Liabilities"] - item["Long Term Debt"] - item["Bonds"] - item["Lease Liabilities"] - item["Retirement Benefit"] - item["Deferred Tax Liabilities"] - item["Asset Retirement Obligation"] - item["Provision NCL True"]
        other_ncl.append(temp)

    df_jasa_ncl.loc["Other Non-Current Liabilities"] = other_ncl

    df_jasa_ncl = df_jasa_ncl.reindex(index=["Total Non-Current Liabilities",
                                            "Long Term Debt",
                                            "Bonds",
                                            "Lease Liabilities",
                                            "Retirement Benefit",
                                            "Deferred Tax Liabilities",
                                            "Asset Retirement Obligation",
                                            "Other Non-Current Liabilities",
                                            "Provision NCL True"])

    return df_jasa_ncl

# 純資産(Net Assets)
def JASA_equity(df_info, df_key_list, filtered_df):
    jasa_equity_code = ["Total Liabilities",
                        "Common Stock",
                        "Total, Additional Paid-In",
                        "Retained Earnings",
                        "Treasury Stock-Common",
                        "Total Shareholder's Equity",
                        "Other Comprehensive Income, Total",
                        "Other Equity, Total",
                        "Non-Controlling Interests",
                        "Net Assets",
                        "Total Liabilities & Net Assets"]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
    join_df = join_df.fillna(0.1)
    df_jasa_equity = join_df[join_df["JASA"].isin(jasa_equity_code)]
    df_jasa_equity = df_jasa_equity.drop_duplicates(subset="要素ID")
    df_jasa_equity = df_jasa_equity.drop("要素ID", axis = 1)
    df_jasa_equity.iloc[:, 1:] = df_jasa_equity.iloc[:, 1:].astype(np.float64)
    df_jasa_equity = df_jasa_equity.groupby(["JASA"]).sum()

    df_jasa_equity = df_jasa_equity.reindex(index=["Total Liabilities",
                                                    "Common Stock",
                                                    "Total, Additional Paid-In",
                                                    "Retained Earnings",
                                                    "Treasury Stock-Common",
                                                    "Other Comprehensive Income, Total",
                                                    "Other Equity, Total",
                                                    "Non-Controlling Interests",
                                                    "Net Assets",
                                                    "Total Liabilities & Net Assets",
                                                    "Total Shareholder's Equity"])

    return df_jasa_equity

# 収入(revenue)
def JASA_revenue(df_info, df_key_list, filtered_df):
    jasa_revenue_code = ["Total Revenue 1",
                        "Total Revenue 2",
                        "Total Revenue 3",
                        "Total Revenue 4",
                        "Total Revenue 5",
                        "Cost of Revenue, Total 1",
                        "Cost of Revenue, Total 2",
                        "Gross Profit",
                        "Gross Profit 3"]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
    join_df = join_df.fillna(0.1)
    df_jasa_revenue = join_df[join_df["JASA"].isin(jasa_revenue_code)]
    df_jasa_revenue = df_jasa_revenue.drop("要素ID", axis = 1)
    df_jasa_revenue.iloc[:, 1:] = df_jasa_revenue.iloc[:, 1:].astype(np.float64)
    df_jasa_revenue = df_jasa_revenue.groupby(["JASA"]).sum()

    revenue_values = []
    for column_name, item in df_jasa_revenue.items():
        if (column_name == "9706" or column_name == "6183" or column_name == "6028" or 
    column_name == "8572" or column_name == "8515" or column_name == "3563" or 
    column_name == "9069" or column_name == "9301" or column_name == "9324" or 
    column_name == "9143" or column_name == "9733" or column_name == "8905" or 
    column_name == "8801" or column_name == "3289" or column_name == "3231" or 
    column_name == "8830" or column_name == "8802" or column_name == "8804" or 
    column_name == "8609" or column_name == "8616" or column_name == "7453" or 
    column_name == "9304" or column_name == "9302" or column_name == "9502" or 
    column_name == "9506" or column_name == "9513" or column_name == "9508" or 
    column_name == "9504" or column_name == "3003"):
            temp = item["Total Revenue 3"]
        elif (column_name == "9602" or column_name == "9206"):
            temp = item["Total Revenue 4"]
        elif (column_name == "6532" or column_name == "8252"):
            temp = item["Total Revenue 2"]
        else:
            temp = item["Total Revenue 1"]
        revenue_values.append(temp)

    df_jasa_revenue.loc["Total Revenue True"] = revenue_values
        
    cost_of_revenue_values = []
    for column_name, item in df_jasa_revenue.items():
        if (column_name == "9602" or column_name == "9069" or column_name == "9301" or 
    column_name == "9324" or column_name == "9143" or column_name == "9733" or 
    column_name == "8905" or column_name == "8801" or column_name == "3289" or 
    column_name == "3231" or column_name == "8830" or column_name == "8802" or 
    column_name == "8804" or column_name == "7453" or column_name == "9304" or 
    column_name == "3003"):
            temp = item["Cost of Revenue, Total 2"]
        else:
            temp = item["Cost of Revenue, Total 1"]
        cost_of_revenue_values.append(temp)

    df_jasa_revenue.loc["Cost of Revenue, Total True"] = cost_of_revenue_values 


    gross_values = []
    for column_name, item in df_jasa_revenue.items():
        if (column_name == "9706" or column_name == "9069" or column_name == "9301" or 
    column_name == "9324" or column_name == "9143" or column_name == "9733" or 
    column_name == "9206" or column_name == "8905" or column_name == "3289" or 
    column_name == "3231" or column_name == "8802" or column_name == "8804" or 
    column_name == "7453" or column_name == "9304" or column_name == "9302" or 
    column_name == "3003"):
            temp = item["Gross Profit 3"]
        else:
            temp = item["Gross Profit"]
        gross_values.append(temp)

    df_jasa_revenue.loc["Gross Profit True"] = gross_values 


    df_jasa_revenue = df_jasa_revenue.reindex(index = ["Total Revenue True",
                                                    "Total Revenue 1",
                                                    "Total Revenue 2",
                                                    "Total Revenue 3",
                                                    "Total Revenue 4",
                                                    "Total Revenue 5",
                                                    "Cost of Revenue, Total True",
                                                "Cost of Revenue, Total 1",
                                                "Cost of Revenue, Total 2",
                                                "Gross Profit True",
                                                "Gross Profit",
                                                "Gross Profit 3"])

    return df_jasa_revenue

# その他PL
def JASA_pl(df_info, df_key_list, filtered_df):
    jasa_pl_code = ["Total Operating Expenses",
                    "Selling/General/Admin. Expenses, Total",
                    "Personnel Expenses",
                    "Personnel Expenses a",
                    "Research & Development",
                    "Advertising Expenses",
                    "Outsourcing Cost",
                    "Rents",
                    "Depreciation / Amortization",
                    "Depreciation / Amortization a",
                    "Logistics Costs 1",
                    "Logistics Costs 2",
                    "Operating Income 1",
                    "Operating Income 2",
                    "Non-Operating Income",
                    "Interest and Dividend Income PL",
                    "Interest and Dividend Income PL 2",
                    "Interest and Dividend Income PL 3",
                    "Investment Gains on Equity Method",
                    "Gains From Foreign Exchange",
                    "Rental Income",
                    "Gains From Sale of Assets",
                    "Subsidies",
                    "Non-Operating Expenses",
                    "Interest Expenses PL",
                    "Interest Expenses PL 2",
                    "Loss From Sale of Assets",
                    "Investment Loss on Equity Method",
                    "Rental Expenses",
                    "Loss From Foreign Exchange",
                    "Transaction Fees",
                    "Ordinary Profit",
                    "Extraordinary Income",
                    "Extraordinary Loss",
                    "Net Income Before Taxes",
                    "Provision for Income Taxes",
                    "Net Income"]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
    #join_df = join_df.dropna(subset = "JASA") #ここを消した場合、別記事業codeが入って、error(float + str)が出るはず。でもここが問題か？
    join_df = join_df.fillna(0)
    df_jasa_pl = join_df[join_df["JASA"].isin(jasa_pl_code)]
    df_jasa_pl = df_jasa_pl.drop_duplicates(subset="要素ID")
    df_jasa_pl = df_jasa_pl.drop("要素ID", axis = 1)
    df_jasa_pl.iloc[:, 1:] = df_jasa_pl.iloc[:, 1:].astype(np.float64)
    df_jasa_pl = df_jasa_pl.groupby(["JASA"]).sum()
    filtered_df.iloc[:, 1:] = filtered_df.iloc[:, 1:].astype(np.float64, errors="ignore")

    interest_pl_values = []
    for column_name, item in df_jasa_pl.items():
        if (item["Interest and Dividend Income PL"] <= -50 or item["Interest and Dividend Income PL"] >= 50) and item["Interest and Dividend Income PL"] < item["Non-Operating Income"]:
            temp = item["Interest and Dividend Income PL"]
        elif (item["Interest and Dividend Income PL 2"] <= -50 or item["Interest and Dividend Income PL 2"] >= 50) and item["Interest and Dividend Income PL 2"] < item["Non-Operating Income"]:
            temp = item["Interest and Dividend Income PL 2"]
        else:
            temp = item["Interest and Dividend Income PL 3"]
        interest_pl_values.append(temp)

    df_jasa_pl.loc["Interest and Dividend Income PL True"] = interest_pl_values

    interest_pl_losses = []
    for column_name, item in df_jasa_pl.items():
        if item["Interest Expenses PL"] <= -50 or item["Interest Expenses PL"] >= 50:
            temp = item["Interest Expenses PL"]
        else:
            temp = item["Interest Expenses PL 2"]
        interest_pl_losses.append(temp)

    df_jasa_pl.loc["Interest Expenses PL True"] = interest_pl_losses

    research_values = []
    for column_name, item in df_jasa_pl.items():
        temp = item["Research & Development"]
        try:
            # ここで何らかの処理を行う
            if column_name == str(2229):
                temp += float(filtered_df.loc["jpcrp030000-asr_E25303-000:ResearchAndDevelopmentExpensesGeneralAndAdministrativeExpenses", str(column_name)])
            elif column_name == str(9783):
                temp += float(filtered_df.loc["jpcrp030000-asr_E04939-000:ResearchAndDevelopmentCostsGeneralAndAdministrativeExpenses", str(column_name)])
                temp += float(filtered_df.loc["jpcrp030000-asr_E04939-000:ResearchAndDevelopmentCostsManufacturingCostForCurrentPeriod", str(column_name)])
        except KeyError:
            print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

        research_values.append(temp)

    df_jasa_pl.loc["Research & Development"] = research_values

    outsourcing_cost = []
    for column_name, item in df_jasa_pl.items():
        temp = item["Outsourcing Cost"]
        try:
            # ここで何らかの処理を行う
            if column_name == str(9470):
                temp += float(filtered_df.loc["jpcrp030000-asr_E00707-000:WorkConsignmentExpenses", str(column_name)])
            elif column_name == str(9202):
                temp += float(filtered_df.loc["jpcrp030000-asr_E04273-000:OutsourcingFeeSGA", str(column_name)])
            elif column_name == str(9532):
                temp += float(filtered_df.loc["jpcrp030000-asr_E04520-000:ConsignmentWorkExpensesSGA", str(column_name)])
            elif column_name == str(9513):
                temp += float(filtered_df.loc["jpcrp030000-asr_E04510-000:ConsignmentCostSGA", str(column_name)])
        except KeyError:
            print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

        outsourcing_cost.append(temp)

    df_jasa_pl.loc["Outsourcing Cost"] = outsourcing_cost

    advertising_cost = []
    for column_name, item in df_jasa_pl.items():
        temp = item["Advertising Expenses"]
        try:
            # ここで何らかの処理を行う
            if column_name == str(2613):
                temp += float(filtered_df.loc["jpcrp030000-asr_E00434-000:AdvertisementSGA", str(column_name)])
            elif column_name == str(2267):
                temp += float(filtered_df.loc["jpcrp030000-asr_E00406-000:SalesPromotionExpensesSGA", str(column_name)])
            elif column_name == str(3382):
                temp += float(filtered_df.loc["jpcrp030000-asr_E03462-000:AdvertisingAndDecorationExpenses", str(column_name)])
            elif column_name == str(4676):
                temp += float(filtered_df.loc["jpcrp030000-asr_E04462-000:AdvertisementSellingExpenses", str(column_name)])
        except KeyError:
            print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

        advertising_cost.append(temp)

    df_jasa_pl.loc["Advertising Expenses"] = advertising_cost

    rent = []
    for column_name, item in df_jasa_pl.items():
        temp = item["Rents"]
        try:
            if column_name == str(9302):
                temp -= float(filtered_df.loc["jppfs_cor:RentExpensesSGA", str(column_name)])
            elif column_name == str(7816):
                temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:Rents", str(column_name)])
        except KeyError:
            print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

        rent.append(temp)

    df_jasa_pl.loc["Rents"] = rent

    depreciation_values = []
    for column_name, item in df_jasa_pl.items():
        if item["Depreciation / Amortization"] <= -50 or item["Depreciation / Amortization"] >= 50:
            temp = item["Depreciation / Amortization"] + item["Depreciation / Amortization a"]
            try:
                if column_name == str(5902):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:AmortizationOfGoodwill", str(column_name)])
                elif column_name == str(6532):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E32549-000:DepreciationAndAmortizationSGA", str(column_name)])
            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

        else:
            temp = item["Depreciation / Amortization a"]
            try:
                if column_name == str(5902):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:AmortizationOfGoodwill", str(column_name)])
                elif column_name == str(6532):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E32549-000:DepreciationAndAmortizationSGA", str(column_name)])

            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

        depreciation_values.append(temp)

    df_jasa_pl.loc["Depreciation / Amortization True"] = depreciation_values 
    
    personnel_values = []
    for column_name, item in df_jasa_pl.items():
        if item["Personnel Expenses"] <= -50 or item["Personnel Expenses"] >= 50:
            temp = item["Personnel Expenses"] + item["Personnel Expenses a"]
            try:
                if column_name == str(7972):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E02371-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E02371-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                elif column_name == str(2168):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E05729-000:EmployeePayrollsAndCompensationAndOtherSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E05729-000:EmployeePayrollsAndCompensationAndOtherSGA", str(column_name)])
                elif column_name == str(9278):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E34102-000:SalariesOfPartTimeEmployeesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E34102-000:SalariesOfPartTimeEmployeesSGA", str(column_name)])
                elif column_name == str(2229):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E25303-000:PayrollsSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E25303-000:PayrollsSGA", str(column_name)])
                elif column_name == str(9861):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03153-000:CostForPartTimersSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E03153-000:CostForPartTimersSGA", str(column_name)])
                elif column_name == str(7581):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03305-000:EmployeesPayrollsAndBonusesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E03305-000:EmployeesPayrollsAndBonusesSGA", str(column_name)])
                elif column_name == str(4934):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E36046-000:RetirementBenefitExpense", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E36046-000:RetirementBenefitExpense", str(column_name)])
                elif column_name == str(4031):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00789-000:PayrollsAllowancesAndBonusesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00789-000:PayrollsAllowancesAndBonusesSGA", str(column_name)])
                elif column_name == str(6753):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01773-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01773-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                elif column_name == str(6367):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01570-000:DirectorsAndEmployeePayrollsAndAllowancesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01570-000:DirectorsAndEmployeePayrollsAndAllowancesSGA", str(column_name)])
                elif column_name == str(9301):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                elif column_name == str(5929):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01385-000:ProvisionForEmployeeCompensationSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01385-000:ProvisionForEmployeeCompensationSGA", str(column_name)])
                elif column_name == str(3407):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00877-000:SalariesAndBenefitsSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                elif column_name == str(3405):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00876-000:PayrollsSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00876-000:PayrollsSGA", str(column_name)])
                elif column_name == str(7269):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E02167-000:PayrollsSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E02167-000:PayrollsSGA", str(column_name)])
                elif column_name == str(8905):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04002-000:ProvisionForDirectorsRemunerationBasedOnPerformanceSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E04002-000:ProvisionForDirectorsRemunerationBasedOnPerformanceSGA", str(column_name)])
                elif column_name == str(2871):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00446-000:DirectorsCompensationEmployeesSalariesBonusesAndAllowancesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00446-000:DirectorsCompensationEmployeesSalariesBonusesAndAllowancesSGA", str(column_name)])
                elif column_name == str(2607):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00431-000:EmployeePayrollsAndAllowances", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00431-000:EmployeePayrollsAndAllowances", str(column_name)])
                elif column_name == str(2004):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00348-000:EmployeeSalariesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00348-000:EmployeeSalariesSGA", str(column_name)])
                elif column_name == str(6332):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                elif column_name == str(6508):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01744-000:BonusesAndProvisionForBonusesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01744-000:BonusesAndProvisionForBonusesSGA", str(column_name)])
                elif column_name == str(6368):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01571-000:EmployeePayrollsAllowancesAndCompensationSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01571-000:EmployeePayrollsAllowancesAndCompensationSGA", str(column_name)])
                elif column_name == str(1301):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:SalesStaffPayrollsAndAllowancesSellingExpenses", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:OfficeStaffPayrollsAndAllowancesGeneralAndAdministrativeExpenses", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                elif column_name == str(5706):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00024-000:BonusAndRetirementPaymentsSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00024-000:BonusAndRetirementPaymentsSGA", str(column_name)])
                elif column_name == str(9304):
                    temp -= float(filtered_df.loc["jppfs_cor:PersonalExpensesSGA", str(column_name)])
                    #print(filtered_df.loc["jppfs_cor:PersonalExpensesSGA", str(column_name)])
                elif column_name == str(9302):
                    temp -= float(filtered_df.loc["jppfs_cor:SalariesAndAllowancesSGA", str(column_name)])
                    #print(filtered_df.loc["jppfs_cor:SalariesAndAllowancesSGA", str(column_name)])
                elif column_name == str(2281):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00335-000:PayrollsAndOtherAllowancesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00335-000:PayrollsAndOtherAllowancesSGA", str(column_name)])
                elif column_name == str(2264):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00331-000:EmployeesSalariesAndBonusesSellingExpenses", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00331-000:EmployeesSalariesAndBonusesGeneralAndAdministrativeExpenses", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                elif column_name == str(2270):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E23202-000:SalariesGeneralAndAdministrativeExpenses", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E23202-000:OtherGeneralAndAdministrativeExpenses", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E23202-000:SalariesGeneralAndAdministrativeExpenses", str(column_name)])
                elif column_name == str(5333):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01137-000:PayrollsAndCompensationSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01137-000:PayrollsAndCompensationSGA", str(column_name)])
                elif column_name == str(3382):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03462-000:SalariesAndWages", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E03462-000:SalariesAndWages", str(column_name)])
                elif column_name == str(5902):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:ProvisionForManagementBoardIncentivePlanTrust", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:ProvisionForEmployeeStockOwnershipPlanTrust", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                elif column_name == str(6861):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01967-000:DirectorsCompensationAndEmployeeSalariesAllowancesAndCompensationSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01967-000:DirectorsCompensationAndEmployeeSalariesAllowancesAndCompensationSGA", str(column_name)])
                elif column_name == str(7816):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:ProvisionForDirectorSPerformanceLinkedIncentiveCompensationSGA", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:ProvisionForEmployeeSPerformanceLinkedIncentiveCompensationSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                elif column_name == str(5233):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01130-000:PersonnelExpenditureSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01130-000:PersonnelExpenditureSGA", str(column_name)])
                elif column_name == str(5232):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01127-000:PayrollsAndBonusesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01127-000:PayrollsAndBonusesSGA", str(column_name)])
                elif column_name == str(4043):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00768-000:SalariesBonusesAndAllowancesSellingExpenses", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00768-000:SalariesBonusesAndAllowancesGeneralAndAdministrativeExpenses", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                elif column_name == str(3191):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E30501-000:PayrollsSGA", str(column_name)])
                elif column_name == str(9206):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E26084-000:PayrollsAndAllowancesSGA", str(column_name)])

                
            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")


        else:
            temp = item["Personnel Expenses a"]
            try:
                if column_name == str(7972):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E02371-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E02371-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                elif column_name == str(2168):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E05729-000:EmployeePayrollsAndCompensationAndOtherSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E05729-000:EmployeePayrollsAndCompensationAndOtherSGA", str(column_name)])
                elif column_name == str(9278):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E34102-000:SalariesOfPartTimeEmployeesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E34102-000:SalariesOfPartTimeEmployeesSGA", str(column_name)])
                elif column_name == str(2229):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E25303-000:PayrollsSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E25303-000:PayrollsSGA", str(column_name)])
                elif column_name == str(9861):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03153-000:CostForPartTimersSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E03153-000:CostForPartTimersSGA", str(column_name)])
                elif column_name == str(7581):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03305-000:EmployeesPayrollsAndBonusesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E03305-000:EmployeesPayrollsAndBonusesSGA", str(column_name)])
                elif column_name == str(4934):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E36046-000:RetirementBenefitExpense", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E36046-000:RetirementBenefitExpense", str(column_name)])
                elif column_name == str(4031):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00789-000:PayrollsAllowancesAndBonusesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00789-000:PayrollsAllowancesAndBonusesSGA", str(column_name)])
                elif column_name == str(6753):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01773-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01773-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                elif column_name == str(6367):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01570-000:DirectorsAndEmployeePayrollsAndAllowancesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01570-000:DirectorsAndEmployeePayrollsAndAllowancesSGA", str(column_name)])
                elif column_name == str(9301):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                elif column_name == str(5929):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01385-000:ProvisionForEmployeeCompensationSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01385-000:ProvisionForEmployeeCompensationSGA", str(column_name)])
                elif column_name == str(3407):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00877-000:SalariesAndBenefitsSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                elif column_name == str(3405):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00876-000:PayrollsSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00876-000:PayrollsSGA", str(column_name)])
                elif column_name == str(7269):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E02167-000:PayrollsSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E02167-000:PayrollsSGA", str(column_name)])
                elif column_name == str(8905):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04002-000:ProvisionForDirectorsRemunerationBasedOnPerformanceSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E04002-000:ProvisionForDirectorsRemunerationBasedOnPerformanceSGA", str(column_name)])
                elif column_name == str(2871):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00446-000:DirectorsCompensationEmployeesSalariesBonusesAndAllowancesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00446-000:DirectorsCompensationEmployeesSalariesBonusesAndAllowancesSGA", str(column_name)])
                elif column_name == str(2607):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00431-000:EmployeePayrollsAndAllowances", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00431-000:EmployeePayrollsAndAllowances", str(column_name)])
                elif column_name == str(2004):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00348-000:EmployeeSalariesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00348-000:EmployeeSalariesSGA", str(column_name)])
                elif column_name == str(6332):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                elif column_name == str(6508):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01744-000:BonusesAndProvisionForBonusesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01744-000:BonusesAndProvisionForBonusesSGA", str(column_name)])
                elif column_name == str(6368):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01571-000:EmployeePayrollsAllowancesAndCompensationSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01571-000:EmployeePayrollsAllowancesAndCompensationSGA", str(column_name)])
                elif column_name == str(1301):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:SalesStaffPayrollsAndAllowancesSellingExpenses", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:OfficeStaffPayrollsAndAllowancesGeneralAndAdministrativeExpenses", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                elif column_name == str(5706):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00024-000:BonusAndRetirementPaymentsSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00024-000:BonusAndRetirementPaymentsSGA", str(column_name)])
                elif column_name == str(9304):
                    temp -= float(filtered_df.loc["jppfs_cor:PersonalExpensesSGA", str(column_name)])
                    #print(filtered_df.loc["jppfs_cor:PersonalExpensesSGA", str(column_name)])
                elif column_name == str(9302):
                    temp -= float(filtered_df.loc["jppfs_cor:SalariesAndAllowancesSGA", str(column_name)])
                    #print(filtered_df.loc["jppfs_cor:SalariesAndAllowancesSGA", str(column_name)])
                elif column_name == str(2281):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00335-000:PayrollsAndOtherAllowancesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E00335-000:PayrollsAndOtherAllowancesSGA", str(column_name)])
                elif column_name == str(2264):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00331-000:EmployeesSalariesAndBonusesSellingExpenses", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00331-000:EmployeesSalariesAndBonusesGeneralAndAdministrativeExpenses", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                elif column_name == str(2270):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E23202-000:SalariesGeneralAndAdministrativeExpenses", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E23202-000:OtherGeneralAndAdministrativeExpenses", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E23202-000:SalariesGeneralAndAdministrativeExpenses", str(column_name)])
                elif column_name == str(5333):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01137-000:PayrollsAndCompensationSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01137-000:PayrollsAndCompensationSGA", str(column_name)])
                elif column_name == str(3382):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03462-000:SalariesAndWages", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E03462-000:SalariesAndWages", str(column_name)])
                elif column_name == str(5902):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:ProvisionForManagementBoardIncentivePlanTrust", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:ProvisionForEmployeeStockOwnershipPlanTrust", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                elif column_name == str(6861):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01967-000:DirectorsCompensationAndEmployeeSalariesAllowancesAndCompensationSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01967-000:DirectorsCompensationAndEmployeeSalariesAllowancesAndCompensationSGA", str(column_name)])
                elif column_name == str(7816):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:ProvisionForDirectorSPerformanceLinkedIncentiveCompensationSGA", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:ProvisionForEmployeeSPerformanceLinkedIncentiveCompensationSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                elif column_name == str(5233):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01130-000:PersonnelExpenditureSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01130-000:PersonnelExpenditureSGA", str(column_name)])
                elif column_name == str(5232):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01127-000:PayrollsAndBonusesSGA", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01127-000:PayrollsAndBonusesSGA", str(column_name)])
                elif column_name == str(4043):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00768-000:SalariesBonusesAndAllowancesSellingExpenses", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00768-000:SalariesBonusesAndAllowancesGeneralAndAdministrativeExpenses", str(column_name)])
                    #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                elif column_name == str(3191):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E30501-000:PayrollsSGA", str(column_name)])
                elif column_name == str(9206):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E26084-000:PayrollsAndAllowancesSGA", str(column_name)])

            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

        personnel_values.append(temp)

    df_jasa_pl.loc["Personnel Expenses True"] = personnel_values


    logistics_costs = []
    for column_name, item in df_jasa_pl.items():
        if item["Logistics Costs 1"] >= 50:
            temp = item["Logistics Costs 1"]
            try:
                if column_name == str(8086):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E02688-000:TransportationExpenses2SGA", str(column_name)])
                elif column_name == str(6367):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01570-000:ProductFreightageExpensesSGA", str(column_name)])
                elif column_name == str(8173):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03052-000:DistributionExpenditureSGA", str(column_name)])
                elif column_name == str(2211):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00374-000:FreightageAndStorageFeeSGA", str(column_name)])
                elif column_name == str(9069):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04179-000:UnderPaymentFareSGA", str(column_name)])
                elif column_name == str(5332):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01138-000:DistributionExpenses", str(column_name)])
                elif column_name == str(5932):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E26831-000:PackingMaterialsAndFreightageExpenditureSGA", str(column_name)])
                elif column_name == str(3405):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00876-000:FreightageAndWarehouseExpensesSGA", str(column_name)])
                elif column_name == str(7269):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E02167-000:ShippingFee", str(column_name)])
                elif column_name == str(2602):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00428-000:ProductsFreightageAndStorageFeeSGA", str(column_name)])
                elif column_name == str(2613):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00434-000:ProductsFreightageFeeSGA", str(column_name)])
                elif column_name == str(2004):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00348-000:ShippingAndDeliveryExpensesSGA", str(column_name)])
                elif column_name == str(1301):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:ShippingAndDeliveryExpensesSellingExpenses", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:StorageExpensesSGA", str(column_name)])
                elif column_name == str(7538):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E02818-000:ShippingBountyAndCompletedDeliveryBountySGA", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E02818-000:FreightExpensesSGA", str(column_name)])
                elif column_name == str(8030):
                    temp += float(filtered_df.loc["jppfs_cor:SalesCommissionSGA", str(column_name)])
                elif column_name == str(7453):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03248-000:DistributionExpensesAndTransportationCostsSGA", str(column_name)])
                elif column_name == str(3315):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00030-000:OceanFreightSGA", str(column_name)])
                elif column_name == str(3106):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00528-000:FreightOutExpenseWarehousingCostAndPackingExpenseSGA", str(column_name)])
                elif column_name == str(2296):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E32069-000:ShippingAndDeliveryExpensesSGA", str(column_name)])
                elif column_name == str(2281):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00335-000:PackingAndFreightageExpensesSGA", str(column_name)])
                elif column_name == str(1383):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E25969-000:PackingAndFreightageExpensesSGA", str(column_name)])
                elif column_name == str(5698):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E27868-000:TransportationSundryExpensesSGA", str(column_name)])
                elif column_name == str(3864):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00644-000:PackingAndFreightageExpensesSGA", str(column_name)])
                elif column_name == str(3865):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00645-000:TransportationExpenditureSGA", str(column_name)])
                elif column_name == str(5233):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01130-000:SalesFreightageExpensesSGA", str(column_name)])
                elif column_name == str(7513):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03270-000:TransportationCostsSGA", str(column_name)])

            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")


        else:
            temp = item["Logistics Costs 2"]
            try:
                if column_name == str(8086):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E02688-000:TransportationExpenses2SGA", str(column_name)])
                elif column_name == str(6367):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01570-000:ProductFreightageExpensesSGA", str(column_name)])
                elif column_name == str(8173):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03052-000:DistributionExpenditureSGA", str(column_name)])
                elif column_name == str(2211):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00374-000:FreightageAndStorageFeeSGA", str(column_name)])
                elif column_name == str(9069):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04179-000:UnderPaymentFareSGA", str(column_name)])
                elif column_name == str(5332):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01138-000:DistributionExpenses", str(column_name)])
                elif column_name == str(5932):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E26831-000:PackingMaterialsAndFreightageExpenditureSGA", str(column_name)])
                elif column_name == str(3405):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00876-000:FreightageAndWarehouseExpensesSGA", str(column_name)])
                elif column_name == str(7269):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E02167-000:ShippingFee", str(column_name)])
                elif column_name == str(2602):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00428-000:ProductsFreightageAndStorageFeeSGA", str(column_name)])
                elif column_name == str(2613):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00434-000:ProductsFreightageFeeSGA", str(column_name)])
                elif column_name == str(2004):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00348-000:ShippingAndDeliveryExpensesSGA", str(column_name)])
                elif column_name == str(1301):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:ShippingAndDeliveryExpensesSellingExpenses", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:StorageExpensesSGA", str(column_name)])
                elif column_name == str(7538):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E02818-000:ShippingBountyAndCompletedDeliveryBountySGA", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E02818-000:FreightExpensesSGA", str(column_name)])
                elif column_name == str(8030):
                    temp += float(filtered_df.loc["jppfs_cor:SalesCommissionSGA", str(column_name)])
                elif column_name == str(7453):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03248-000:DistributionExpensesAndTransportationCostsSGA", str(column_name)])
                elif column_name == str(3315):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00030-000:OceanFreightSGA", str(column_name)])
                elif column_name == str(3106):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00528-000:FreightOutExpenseWarehousingCostAndPackingExpenseSGA", str(column_name)])
                elif column_name == str(2296):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E32069-000:ShippingAndDeliveryExpensesSGA", str(column_name)])
                elif column_name == str(2281):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00335-000:PackingAndFreightageExpensesSGA", str(column_name)])
                elif column_name == str(1383):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E25969-000:PackingAndFreightageExpensesSGA", str(column_name)])
                elif column_name == str(5698):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E27868-000:TransportationSundryExpensesSGA", str(column_name)])
                elif column_name == str(3864):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00644-000:PackingAndFreightageExpensesSGA", str(column_name)])
                elif column_name == str(3865):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00645-000:TransportationExpenditureSGA", str(column_name)])
                elif column_name == str(5233):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01130-000:SalesFreightageExpensesSGA", str(column_name)])
                elif column_name == str(7513):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03270-000:TransportationCostsSGA", str(column_name)])

            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

        logistics_costs.append(temp)

    df_jasa_pl.loc["Logistics Costs True"] = logistics_costs

    operating_values = []
    for column_name, item in df_jasa_pl.items():
        if item["Operating Income 1"] <= -50 or item["Operating Income 1"] >= 50:
            temp = item["Operating Income 1"]
        else :
            temp = item["Operating Income 2"]
        operating_values.append(temp)  

    df_jasa_pl.loc["Operating Income True"] = operating_values  

    other_sga = []
    for column_name, item in df_jasa_pl.items():
        temp = item["Selling/General/Admin. Expenses, Total"] - item["Logistics Costs True"] - item["Personnel Expenses True"] - item["Advertising Expenses"] - item["Outsourcing Cost"] - item["Rents"] - item["Research & Development"] - item["Depreciation / Amortization True"]
        other_sga.append(temp)
    df_jasa_pl.loc["Other Selling/General/Admin. Expenses"] = other_sga

    other_noi = []
    for column_name, item in df_jasa_pl.items():
        temp = item["Non-Operating Income"] - item["Interest and Dividend Income PL True"] - item["Investment Gains on Equity Method"] - item["Gains From Foreign Exchange"] - item["Rental Income"] - item["Subsidies"]
        other_noi.append(temp)
    df_jasa_pl.loc["Other Non-Operating Income"] = other_noi

    other_noe = []
    for column_name, item in df_jasa_pl.items():
        temp = item["Non-Operating Expenses"] - item["Interest Expenses PL True"] - item["Investment Loss on Equity Method"] - item["Loss From Foreign Exchange"] - item["Rental Expenses"] - item["Transaction Fees"]
        other_noe.append(temp)
    df_jasa_pl.loc["Other Non-Operating Expenses"] = other_noe

    df_jasa_pl = df_jasa_pl.reindex(index=["Total Operating Expenses",
                                                "Selling/General/Admin. Expenses, Total",
                                                "Logistics Costs True",
                                                "Personnel Expenses True",
                                                "Advertising Expenses",
                                                "Outsourcing Cost",
                                                "Rents",
                                                "Research & Development",
                                                "Depreciation / Amortization True",
                                                "Other Selling/General/Admin. Expenses",
                                                "Operating Income True",
                                                "Non-Operating Income",
                                                "Interest and Dividend Income PL True",
                                                "Investment Gains on Equity Method",
                                                "Gains From Foreign Exchange",
                                                "Rental Income",
                                                "Subsidies",
                                                "Other Non-Operating Income",
                                                "Non-Operating Expenses",
                                                "Interest Expenses PL True",
                                                "Investment Loss on Equity Method",
                                                "Loss From Foreign Exchange",
                                                "Rental Expenses",
                                                "Transaction Fees",
                                                "Other Non-Operating Expenses",
                                                "Ordinary Profit",
                                                "Extraordinary Income",
                                                "Extraordinary Loss",
                                                "Net Income Before Taxes",
                                                "Provision for Income Taxes",
                                                "Net Income"])

    return df_jasa_pl

# 営業CF
def JASA_opcf(df_info, df_key_list, filtered_df):
    jasa_opcf_code = ["Net Income Before Taxes",
                    "Depreciation/Depletion",
                    "Amortization of Goodwill",
                    "Depreciation and Amortization on Other",
                    "Increase (Decrease) in Provision",
                    "Interest and Dividend Income CF",
                    "Interest Expenses CF",
                    "Share of Loss (Profit) of Entities Accounted for Using Equity Method",
                    "Other Loss (Gain)",
                    "Decrease (Increase) in Trade Receivables",
                    "Decrease (Increase) in Inventories",
                    "Increase (Decrease) in Advances Received",
                    "Decrease (Increase) in Inventories",
                    "Increase (Decrease) in Advances Received",
                    "Decrease (Increase) in Prepaid Expenses",
                    "Increase (Decrease) in Trade Payables",
                    "Increase (Decrease) in Retirement Benefit Liabilities",
                    "Increase (Decrease) in Accounts Payable - Other, and Accrued Expenses",
                    "Subtotal",
                    "Interest and Dividends Received",
                    "Interest Paid",
                    "Proceeds from Insurance Income",
                    "Compensation Paid for Damage",
                    "Cash Taxes Paid",
                    "Cash from Operating Activities"]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
    #join_df = join_df.dropna(subset = "JASA") 
    join_df = join_df.fillna(0.1)
    df_jasa_opcf = join_df[join_df["JASA"].isin(jasa_opcf_code)]
    df_jasa_opcf = df_jasa_opcf.drop("要素ID", axis = 1)
    df_jasa_opcf.iloc[:, 1:] = df_jasa_opcf.iloc[:, 1:].astype(np.float64)
    df_jasa_opcf = df_jasa_opcf.groupby(["JASA"]).sum()

    other_opcf = []
    for column_name, item in df_jasa_opcf.items():
        temp = item["Subtotal"]

        items_to_subtract = ["Net Income Before Taxes",
            "Depreciation/Depletion",
            "Amortization of Goodwill",
            "Depreciation and Amortization on Other",
            "Increase (Decrease) in Provision",
            "Interest and Dividend Income CF",
            "Interest Expenses CF",
            "Share of Loss (Profit) of Entities Accounted for Using Equity Method",
            "Other Loss (Gain)",
            "Decrease (Increase) in Trade Receivables",
            "Decrease (Increase) in Inventories",
            "Increase (Decrease) in Advances Received",
            "Decrease (Increase) in Prepaid Expenses",
            "Increase (Decrease) in Trade Payables",
            "Increase (Decrease) in Retirement Benefit Liabilities",
            "Increase (Decrease) in Accounts Payable - Other, and Accrued Expenses"
        ]
        for key in items_to_subtract:
            temp -= item[key]
        other_opcf.append(temp)

    df_jasa_opcf.loc["Other, Net"] = other_opcf


    other_pr = []
    for column_name, item in df_jasa_opcf.items():
        temp = item["Cash from Operating Activities"]
        # 引く項目のリスト
        items_to_subtract = [
            "Subtotal",
            "Interest and Dividends Received",
            "Interest Paid",
            "Proceeds from Insurance Income",
            "Compensation Paid for Damage",
            "Cash Taxes Paid"
        ]
        for key in items_to_subtract:
            temp -= item[key]
        other_pr.append(temp)
    
    df_jasa_opcf.loc["Other Cash from Operating Activities"] = other_pr

    df_jasa_opcf = df_jasa_opcf.rename(index={"Net Income Before Taxes":"Net Income/Starting Line"})

    return df_jasa_opcf

# 営業CFの直接法ver.
def JASA_opcf2(df_info, df_key_list, filtered_df):
    jasa_opcf2_code = ["Receipts from Operating Revenue",
                        "Payments for Raw Materials and Goods",
                        "Payments of Personnel Expenses",
                        "Subtotal",
                        "Interest and Dividends Received",
                        "Interest Paid",
                        "Proceeds from Insurance Income",
                        "Compensation Paid for Damage",
                        "Cash Taxes Paid",
                        "Cash from Operating Activities"]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
    #join_df = join_df.dropna(subset = "JASA") 
    join_df = join_df.fillna(0.1)
    df_jasa_opcf2 = join_df[join_df["JASA"].isin(jasa_opcf2_code)]
    df_jasa_opcf2 = df_jasa_opcf2.drop("要素ID", axis = 1)
    df_jasa_opcf2.iloc[:, 1:] = df_jasa_opcf2.iloc[:, 1:].astype(np.float64)
    df_jasa_opcf2 = df_jasa_opcf2.groupby(["JASA"]).sum()
    
    other_opcf2 = []
    for column_name, item in df_jasa_opcf2.items():
        temp = item["Subtotal"]
        # 引く項目のリスト
        items_to_subtract = [
            "Receipts from Operating Revenue",
            "Payments for Raw Materials and Goods",
            "Payments of Personnel Expenses"
        ]
        for key in items_to_subtract:
            temp -= item[key]
        other_opcf2.append(temp)
    
    df_jasa_opcf2.loc["Other, Net"] = other_opcf2

    other_pr2 = []
    for column_name, item in df_jasa_opcf2.items():
        temp = item["Cash from Operating Activities"]
        # 引く項目のリスト
        items_to_subtract = [
            "Subtotal",
            "Interest and Dividends Received",
            "Interest Paid",
            "Proceeds from Insurance Income",
            "Compensation Paid for Damage",
            "Cash Taxes Paid"
        ]
        for key in items_to_subtract:
            temp -= item[key]
        other_pr2.append(temp)

    df_jasa_opcf2.loc["Other Cash from Operating Activities"] = other_pr2

    return df_jasa_opcf2

# 投資CF
def JASA_incf(df_info, df_key_list, filtered_df):
    jasa_incf_code = ["Purchase of Securities",
                        "Proceeds from Sale of Securities",
                        "Purchase of Property, Plant and Equipment",
                        "Proceeds from Sale of Property, Plant and Equipment",
                        "Purchase of Intangible Assets",
                        "Proceeds from Sale of Intangible Assets",
                        "Purchase of Investment Securities",
                        "Proceeds from Sale of Investment Securities",
                        "Purchase of Shares of Subsidiaries Resulting in Change in Scope of Consolidation",
                        "Proceeds from Sale of Shares of Subsidiaries Resulting in Change in Scope of Consolidation",
                        "Loans Advances",
                        "Proceeds from Collection of Loans Receivable",
                        "Payments into Time Deposits",
                        "Proceeds from Withdrawal of Time Deposits",
                        "Payments of Leasehold and Guarantee Deposits",
                        "Proceeds from Refund of Leasehold and Guarantee Deposits",
                        "Payments for Acquisition of Businesses",
                        "Proceeds from Sale of Businesses",
                        "Cash from Investing Activities"]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
    #join_df = join_df.dropna(subset = "JASA") 
    join_df = join_df.fillna(0.1)
    df_jasa_incf = join_df[join_df["JASA"].isin(jasa_incf_code)]
    df_jasa_incf = df_jasa_incf.drop("要素ID", axis = 1)
    df_jasa_incf.iloc[:, 1:] = df_jasa_incf.iloc[:, 1:].astype(np.float64)
    df_jasa_incf = df_jasa_incf.groupby(["JASA"]).sum()

    other_incf = []
    for column_name, item in df_jasa_incf.items():
        temp = item["Cash from Investing Activities"]
        # 引く項目のリスト
        items_to_subtract = [
            "Purchase of Securities",
            "Proceeds from Sale of Securities",
            "Purchase of Property, Plant and Equipment",
            "Proceeds from Sale of Property, Plant and Equipment",
            "Purchase of Intangible Assets",
            "Proceeds from Sale of Intangible Assets",
            "Purchase of Investment Securities",
            "Proceeds from Sale of Investment Securities",
            "Purchase of Shares of Subsidiaries Resulting in Change in Scope of Consolidation",
            "Proceeds from Sale of Shares of Subsidiaries Resulting in Change in Scope of Consolidation",
            "Loans Advances",
            "Proceeds from Collection of Loans Receivable",
            "Payments into Time Deposits",
            "Proceeds from Withdrawal of Time Deposits",
            "Payments of Leasehold and Guarantee Deposits",
            "Proceeds from Refund of Leasehold and Guarantee Deposits",
            "Payments for Acquisition of Businesses",
            "Proceeds from Sale of Businesses"
        ]
        for key in items_to_subtract:
            temp -= item[key]
        other_incf.append(temp)
    
    df_jasa_incf.loc["Other Cash from Investing Activities"] = other_incf

    return df_jasa_incf

# 財務CF
def JASA_ficf(df_info, df_key_list, filtered_df):
    jasa_ficf_code= [
                        "Proceeds from Short Term Borrowings",
                        "Repayments of Short Term Borrowings",
                        "Proceeds from Long Term Borrowings",
                        "Repayments of Long Term Borrowings",
                        "Proceeds from Issuance of Bonds",
                        "Redemption of Bonds",
                        "Proceeds from Share Issuance to Non-Controlling Shareholders",
                        "Proceeds from Sale of Shares of Subsidiaries Not Resulting in Change in Scope of Consolidation",
                        "Purchase of Shares of Subsidiaries Not Resulting in Change in Scope of Consolidation",
                        "Purchase of Treasury Shares",
                        "Proceeds from Sale of Treasury Shares",
                        "Repayments of Lease Liabilities",
                        "Dividends Paid",
                        "Dividends Paid to Non-Controlling Interests",
                        "Cash from Financing Activities"]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
    #join_df = join_df.dropna(subset = "JASA") 
    join_df = join_df.fillna(0.1)
    df_jasa_ficf = join_df[join_df["JASA"].isin(jasa_ficf_code)]
    df_jasa_ficf = df_jasa_ficf.drop("要素ID", axis = 1)
    df_jasa_ficf.iloc[:, 1:] = df_jasa_ficf.iloc[:, 1:].astype(np.float64)
    df_jasa_ficf = df_jasa_ficf.groupby(["JASA"]).sum()

    other_ficf = []
    for column_name, item in df_jasa_ficf.items():
        temp = item["Cash from Financing Activities"]
        # 引く項目のリスト
        items_to_subtract = [
            "Proceeds from Short Term Borrowings",
            "Repayments of Short Term Borrowings",
            "Proceeds from Long Term Borrowings",
            "Repayments of Long Term Borrowings",
            "Proceeds from Issuance of Bonds",
            "Redemption of Bonds",
            "Proceeds from Share Issuance to Non-Controlling Shareholders",
            "Proceeds from Sale of Shares of Subsidiaries Not Resulting in Change in Scope of Consolidation",
            "Purchase of Shares of Subsidiaries Not Resulting in Change in Scope of Consolidation",
            "Purchase of Treasury Shares",
            "Proceeds from Sale of Treasury Shares",
            "Repayments of Lease Liabilities",
            "Dividends Paid",
            "Dividends Paid to Non-Controlling Interests"]
        
        for key in items_to_subtract:
            temp -= item[key]
        other_ficf.append(temp)
    
    df_jasa_ficf.loc["Other Cash from Financing Activities"] = other_ficf

    return df_jasa_ficf


def JASA_date(df_info, df_key_list):
    jasa_date_code = ["jpdei_cor:CurrentPeriodEndDateDEI"]
    join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
    #join_df = join_df.dropna(subset = "JASA") #ここを消した場合、別記事業codeが入って、error(float + str)が出るはず。でもここが問題か？
    join_df = join_df.fillna(0.1)
    df_jasa_date = join_df[join_df["要素ID"].isin(jasa_date_code)]
    df_jasa_date = df_jasa_date.drop_duplicates(subset="要素ID")
    df_jasa_date = df_jasa_date.drop("要素ID", axis = 1)
    #df_jasa_date.iloc[:, 1:] = df_jasa_date.iloc[:, 1:].astype(np.float64)
    #df_jasa_equity = df_jasa_equity.groupby(["JASA"]).sum()

    return df_jasa_date


# 全ての情報を一括取得して、JASAフォーマットに変換する(use)
def JASA_all(df_info, df_key_list, filtered_df):
    # 流動資産(Current Assets)
    def JASA_ca(df_info, df_key_list, filtered_df):
        jasa_ca_code = ["Cash and Short Term Investments", 
                    "Notes Receivable/Accounts Receivable",
                "Notes Receivable/Accounts Receivable 1",
                "Notes Receivable/Accounts Receivable 2",
                "Notes Receivable/Accounts Receivable 3",
                "Notes Receivable/Accounts Receivable 4",
                "Notes Receivable/Accounts Receivable 5",
                "Notes Receivable/Accounts Receivable 6",
                "Notes Receivable/Accounts Receivable 7",
                "Electronically Recorded Monetary Claims",
                "Total Inventory",
                "Inventory1", 
                "Inventory2",
                "Inventory3",
                "Inventory4",
                "Inventory",
                "Prepaid Expenses",
                "Trading Securities",
                "Total Current Assets"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") 
        join_df = join_df.fillna(0.1)
        df_jasa_ca = join_df[join_df["JASA"].isin(jasa_ca_code)]
        df_jasa_ca = df_jasa_ca.drop("要素ID", axis = 1)
        df_jasa_ca.iloc[:, 1:] = df_jasa_ca.iloc[:, 1:].astype(np.float64)
        df_jasa_ca = df_jasa_ca.groupby(["JASA"]).sum()

        inventory_values = []
        for column_name, item in df_jasa_ca.items():
            if item["Inventory1"] > 1:
                if item["Inventory3"] > 1:
                    temp = item["Inventory1"] + item["Inventory3"] + item["Inventory"]
                else:
                    temp = item["Inventory1"] + item["Inventory4"] + item["Inventory"]
            else:
                if item["Inventory3"] > 1:
                    temp = item["Inventory2"] + item["Inventory3"] + item["Inventory"]
                else:
                    temp = item["Inventory2"] + item["Inventory4"] + item["Inventory"]
            inventory_values.append(temp)

        df_jasa_ca.loc["Inventory True"] = inventory_values 

        #数値が大きい方を取ってきている。
        total_inventory_values = []
        for column_name, item in df_jasa_ca.items():
            if item["Total Inventory"] > item["Inventory True"]: 
                temp = item["Total Inventory"]
                if column_name == str(1802):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00055-000:InventoriesForPFIAndOtherProjectsCA", str(column_name)])
                elif column_name == str(1803):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:UncompletedRealEstateDevelopmentProjectsCA", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:PFIProjectsAndOtherInventoriesCA", str(column_name)])
                elif column_name == str(1812):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00058-000:CostsOnDevelopmentProjectsInProgressCA", str(column_name)])
            else:
                temp = item["Inventory True"]
                if column_name == str(1802):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00055-000:InventoriesForPFIAndOtherProjectsCA", str(column_name)])
                elif column_name == str(1803):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:UncompletedRealEstateDevelopmentProjectsCA", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:PFIProjectsAndOtherInventoriesCA", str(column_name)])
                elif column_name == str(1812):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00058-000:CostsOnDevelopmentProjectsInProgressCA", str(column_name)])

            total_inventory_values.append(temp)

        df_jasa_ca.loc["Total Inventory True"] = total_inventory_values

        receivable_values = []
        for column_name, item in df_jasa_ca.items():
            if item["Notes Receivable/Accounts Receivable 1"] > 1:
                temp = item["Notes Receivable/Accounts Receivable 1"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            elif item["Notes Receivable/Accounts Receivable 2"] > 1:
                temp = item["Notes Receivable/Accounts Receivable 2"] + item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            elif item["Notes Receivable/Accounts Receivable 3"] > 1:
                temp = item["Notes Receivable/Accounts Receivable 3"] + item["Notes Receivable/Accounts Receivable 4"] + item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            elif item["Notes Receivable/Accounts Receivable 6"] > 1:
                temp = item["Notes Receivable/Accounts Receivable 6"] + item["Notes Receivable/Accounts Receivable 4"] + item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            elif item["Notes Receivable/Accounts Receivable 4"] > 1:
                temp = item["Notes Receivable/Accounts Receivable 4"] + item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            elif item["Notes Receivable/Accounts Receivable 5"] > 1:
                temp = item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            else:
                temp = item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            receivable_values.append(temp)

        df_jasa_ca.loc["Notes Receivable/Accounts Receivable True"] = receivable_values

        # receivable_ca = []
        # for column_name, item in df_jasa_ca.items():
        #     temp = item["Notes Receivable/Accounts Receivable True"]
        #     if column_name == str(6532):
        #         temp += float(filtered_df.loc["jpcrp030000-asr_E32549-000:AccountsReceivableTradeAndContractAssetsCA", str(column_name)])
        #     receivable_ca.append(temp)
        
        # df_jasa_ca.loc["Notes Receivable/Accounts Receivable True"] = receivable_ca

        other_ca = []
        for column_name, item in df_jasa_ca.items():
            temp = item["Total Current Assets"] - item["Cash and Short Term Investments"] - item["Notes Receivable/Accounts Receivable True"] - item["Total Inventory True"] - item["Prepaid Expenses"] - item["Trading Securities"]
            other_ca.append(temp)

        
        df_jasa_ca.loc["Other Current Assets"] = other_ca

        df_jasa_ca = df_jasa_ca.reindex(index=["Total Current Assets", 
                                            "Cash and Short Term Investments",
                                            "Notes Receivable/Accounts Receivable True",
                                            "Total Inventory True",
                                            "Prepaid Expenses",
                                            "Trading Securities",
                                            "Other Current Assets",
                                            "Electronically Recorded Monetary Claims",
                                            "Notes Receivable/Accounts Receivable 1",
                                            "Notes Receivable/Accounts Receivable 2",
                                            "Notes Receivable/Accounts Receivable 3",
                                            "Notes Receivable/Accounts Receivable 4",
                                            "Notes Receivable/Accounts Receivable 5",
                                            "Notes Receivable/Accounts Receivable 6",
                                            "Notes Receivable/Accounts Receivable 7",
                                            "Notes Receivable/Accounts Receivable"])

        return df_jasa_ca

    # 固定資産(Non-Current Assets)
    def JASA_nca(df_info, df_key_list, filtered_df):
        jasa_nca_code = ["Total Non-Current Assets",
                        "Property/Plant/Equipment, Total - Net",
                        "Intangibles, Net",
                        "Goodwill, Net",
                        "Investments and Other Assets",
                        "Long Term Investments",
                        "Leasehold and Guarantee Deposits 1",
                        "Leasehold and Guarantee Deposits 2",
                        "Long Term Deposits",
                        "Deferred Tax Assets",
                        "Total Assets"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") #ここを消した場合、別記事業codeが入って、error(float + str)が出るはず。でもここが問題か？
        join_df = join_df.fillna(0.1)
        df_jasa_nca = join_df[join_df["JASA"].isin(jasa_nca_code)]
        df_jasa_nca = df_jasa_nca.drop("要素ID", axis = 1)
        df_jasa_nca.iloc[:, 1:] = df_jasa_nca.iloc[:, 1:].astype(np.float64)
        df_jasa_nca = df_jasa_nca.groupby(["JASA"]).sum()

        lease_values = []
        for column_name, item in df_jasa_nca.items():
            if item["Leasehold and Guarantee Deposits 1"] > 1:
                temp = item["Leasehold and Guarantee Deposits 1"]
            else:
                temp = item["Leasehold and Guarantee Deposits 2"]
            lease_values.append(temp)

        df_jasa_nca.loc["Leasehold and Guarantee Deposits True"] = lease_values

        intangibles_other_values = []
        for column_name, item in df_jasa_nca.items():
            temp = item["Intangibles, Net"] - item["Goodwill, Net"]
            intangibles_other_values.append(temp)

        df_jasa_nca.loc["Other Intangibles, Net"] = intangibles_other_values

        other_ncl = []
        for column_name, item in df_jasa_nca.items():
            temp = item["Total Non-Current Assets"] - item["Property/Plant/Equipment, Total - Net"] - item["Goodwill, Net"] - item["Other Intangibles, Net"] - item["Long Term Investments"] - item["Leasehold and Guarantee Deposits True"] - item["Deferred Tax Assets"]
            other_ncl.append(temp)
        
        df_jasa_nca.loc["Other Non-Current Assets"] = other_ncl

        df_jasa_nca = df_jasa_nca.reindex(index=["Total Non-Current Assets",
                                                "Property/Plant/Equipment, Total - Net",
                                                "Goodwill, Net",
                                                "Other Intangibles, Net",
                                                "Long Term Investments",
                                                "Leasehold and Guarantee Deposits True",
                                                "Deferred Tax Assets",
                                                "Other Non-Current Assets",
                                                "Total Assets"])

        return df_jasa_nca

    # 流動負債(Current Liabilities)
    def JASA_cl(df_info, df_key_list, filtered_df):
        jasa_cl_code = ["Notes Payable/Accounts Payable",
                        "Notes Payable/Accounts Payable 1",
                    "Notes Payable/Accounts Payable 2",
                    "Notes Payable/Accounts Payable 3",
                    "Notes Payable/Accounts Payable 4",
                    "Notes Payable/Accounts Payable 5",
                    "Notes Payable/Accounts Payable 6",
                    "Notes Payable/Accounts Payable 7",
                    "Electronically Recorded Monetary Debt",
                    "Capital Leases",
                    "Short Term Debt",
                    "Current Port. of LT Debt",
                    "Advances Received",
                    "Total Current Liabilities",
                    "Provision CL",
                    "Provision CL, Total"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        join_df = join_df.fillna(0.1)
        df_jasa_cl = join_df[join_df["JASA"].isin(jasa_cl_code)]
        df_jasa_cl = df_jasa_cl.drop("要素ID", axis = 1)
        df_jasa_cl.iloc[:, 1:] = df_jasa_cl.iloc[:, 1:].astype(np.float64)
        df_jasa_cl = df_jasa_cl.groupby(["JASA"]).sum()

        notes_payable_values = []
        for column_name, item in df_jasa_cl.items():
            if item["Notes Payable/Accounts Payable 1"] > 1:
                if item["Notes Payable/Accounts Payable 6"] > 1:
                    temp = item["Notes Payable/Accounts Payable 1"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
                else:
                    temp = item["Notes Payable/Accounts Payable 1"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
            elif item["Notes Payable/Accounts Payable 2"]:
                if item["Notes Payable/Accounts Payable 6"] > 1:
                    temp = item["Notes Payable/Accounts Payable 2"] + item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
                else:
                    temp = item["Notes Payable/Accounts Payable 2"] + item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
            elif item["Notes Payable/Accounts Payable 5"] > 1:
                if item["Notes Payable/Accounts Payable 6"] > 1:
                    temp = item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 5"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
                else:
                    temp = item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 5"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
            elif item["Notes Payable/Accounts Payable 3"] > 1:
                if item["Notes Payable/Accounts Payable 6"] > 1:
                    temp = item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
                elif item["Notes Payable/Accounts Payable 7"] > 1:
                    temp = item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
            else:
                if item["Notes Payable/Accounts Payable 6"] > 1:
                    temp = item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
                else:
                    temp = item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
            notes_payable_values.append(temp)

        df_jasa_cl.loc["Notes Payable/Accounts Payable True"] = notes_payable_values  

        # advances_values = []
        # for column_name, item in df_jasa_cl.items():
        #     if item["Advances Received 2"] > 1:
        #         temp = item["Advances Received 2"]
        #     else:
        #         temp = item["Advances Received 1"]
        #     advances_values.append(temp)

        # df_jasa_cl.loc["Advances Received True"] = advances_values

        advances_values = []
        for column_name, item in df_jasa_cl.items():
            temp = item["Advances Received"]
            if column_name == str(1802):
                temp += float(filtered_df.loc["jpcrp030000-asr_E00055-000:AdvancesReceivedOnConstructionContractsInProgressContractLiabilitiesCLCNS", str(column_name)])
            elif column_name == str(1812):
                temp += float(filtered_df.loc["jpcrp030000-asr_E00058-000:AdvancesReceivedOnConstructionProjectsInProgress", str(column_name)])
            advances_values.append(temp)
    
        df_jasa_cl.loc["Advances Received True"] = advances_values

        short_values = []
        for column_name, item in df_jasa_cl.items():
            if item["Short Term Debt"] < item["Total Current Liabilities"] - (item["Notes Payable/Accounts Payable True"] + item["Advances Received True"] + item["Capital Leases"]):
                temp = item["Short Term Debt"]
            else:
                temp = 0
            short_values.append(temp)

        df_jasa_cl.loc["Short Term Debt"] = short_values

        long_values = []
        for column_name, item in df_jasa_cl.items():
            if item["Current Port. of LT Debt"] < item["Total Current Liabilities"] - (item["Notes Payable/Accounts Payable True"] + item["Advances Received True"] + item["Capital Leases"] + item["Short Term Debt"]):
                temp = item["Current Port. of LT Debt"]
                if column_name == str(1802):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00055-000:CurrentPortionOfNonrecourseLoansCL", str(column_name)])
                elif column_name == str(1803):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:CurrentPortionOfNonRecourseBorrowingsCL", str(column_name)])
            else:
                temp = 0
                if column_name == str(1802):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00055-000:CurrentPortionOfNonrecourseLoansCL", str(column_name)])
                elif column_name == str(1803):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:CurrentPortionOfNonRecourseBorrowingsCL", str(column_name)])
            long_values.append(temp)

        df_jasa_cl.loc["Current Port. of LT Debt"] = long_values

        provision_cl = []
        for column_name, item in df_jasa_cl.items():
            if item["Provision CL, Total"] > 1:
                temp = item["Provision CL, Total"]
            else:
                temp = item["Provision CL"]
            provision_cl.append(temp)
        
        df_jasa_cl.loc["Provision CL True"] = provision_cl

        other_cl = []
        for column_name, item in df_jasa_cl.items():
            temp = item["Total Current Liabilities"] - item["Notes Payable/Accounts Payable True"] - item["Advances Received True"] - item["Capital Leases"] - item["Short Term Debt"] - item["Current Port. of LT Debt"] - item["Provision CL True"]
            other_cl.append(temp)
        
        df_jasa_cl.loc["Other Current Liabilities"] = other_cl

        df_jasa_cl = df_jasa_cl.reindex(index=["Total Current Liabilities", 
                                            "Notes Payable/Accounts Payable True",
                                            "Advances Received True",
                                            "Capital Leases",
                                            "Short Term Debt",
                                            "Current Port. of LT Debt",
                                            "Other Current Liabilities",
                                            "Provision CL True",
                                            "Electronically Recorded Monetary Debt",
                                            "Notes Payable/Accounts Payable 1",
                                            "Notes Payable/Accounts Payable 2",
                                            "Notes Payable/Accounts Payable 3",
                                            "Notes Payable/Accounts Payable 4",
                                            "Notes Payable/Accounts Payable 5",
                                            "Notes Payable/Accounts Payable 6",
                                            "Notes Payable/Accounts Payable 7",
                                            "Notes Payable/Accounts Payable"]) 
        
        return df_jasa_cl

    # 固定負債(Non-Current Liabilities)
    def JASA_ncl(df_info, df_key_list, filtered_df):
        jasa_ncl_code = ["Total Non-Current Liabilities",
                        "Long Term Debt",
                        "Lease Liabilities",
                        "Retirement Benefit",
                        "Deferred Tax Liabilities",
                        "Asset Retirement Obligation",
                        "Bonds",
                        "Provision NCL, Total",
                        "Provision NCL"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        join_df = join_df.fillna(0.1)
        df_jasa_ncl = join_df[join_df["JASA"].isin(jasa_ncl_code)]
        df_jasa_ncl = df_jasa_ncl.drop("要素ID", axis = 1)
        df_jasa_ncl.iloc[:, 1:] = df_jasa_ncl.iloc[:, 1:].astype(np.float64)
        df_jasa_ncl = df_jasa_ncl.groupby(["JASA"]).sum()

        provision_ncl = []
        for column_name, item in df_jasa_ncl.items():
            if item["Provision NCL, Total"] > 1:
                temp = item["Provision NCL, Total"]
            else:
                temp = item["Provision NCL"]
            provision_ncl.append(temp)
        
        df_jasa_ncl.loc["Provision NCL True"] = provision_ncl

        long_term_debt = []
        for column_name, item in df_jasa_ncl.items():
            temp = item["Long Term Debt"]
            if column_name == str(1802):
                temp += float(filtered_df.loc["jpcrp030000-asr_E00055-000:NonrecourseLoansNCL", str(column_name)])
            elif column_name == str(1803):
                temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:NonRecourseBorrowingsNCL", str(column_name)])
            long_term_debt.append(temp)

        df_jasa_ncl.loc["Long Term Debt"] = long_term_debt

        other_ncl = []
        for column_name, item in df_jasa_ncl.items():
            temp = item["Total Non-Current Liabilities"] - item["Long Term Debt"] - item["Bonds"] - item["Lease Liabilities"] - item["Retirement Benefit"] - item["Deferred Tax Liabilities"] - item["Asset Retirement Obligation"] - item["Provision NCL True"]
            other_ncl.append(temp)

        df_jasa_ncl.loc["Other Non-Current Liabilities"] = other_ncl

        df_jasa_ncl = df_jasa_ncl.reindex(index=["Total Non-Current Liabilities",
                                                "Long Term Debt",
                                                "Bonds",
                                                "Lease Liabilities",
                                                "Retirement Benefit",
                                                "Deferred Tax Liabilities",
                                                "Asset Retirement Obligation",
                                                "Other Non-Current Liabilities",
                                                "Provision NCL True"])

        return df_jasa_ncl

    # 純資産(Net Assets)
    def JASA_equity(df_info, df_key_list, filtered_df):
        jasa_equity_code = ["Total Liabilities",
                            "Common Stock",
                            "Total, Additional Paid-In",
                            "Retained Earnings",
                            "Treasury Stock-Common",
                            "Total Shareholder's Equity",
                            "Valuation/Exchange Differences, etc.",
                            "Non-Controlling Interests",
                            "Net Assets",
                            "Total Liabilities & Net Assets"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        join_df = join_df.fillna(0.1)
        df_jasa_equity = join_df[join_df["JASA"].isin(jasa_equity_code)]
        df_jasa_equity = df_jasa_equity.drop_duplicates(subset="要素ID")
        df_jasa_equity = df_jasa_equity.drop("要素ID", axis = 1)
        df_jasa_equity.iloc[:, 1:] = df_jasa_equity.iloc[:, 1:].astype(np.float64)
        df_jasa_equity = df_jasa_equity.groupby(["JASA"]).sum()

        other_equity = []
        for column_name, item in df_jasa_equity.items():
            temp = item["Net Assets"] - item["Common Stock"] - item["Total, Additional Paid-In"] - item["Retained Earnings"] - item["Treasury Stock-Common"] - item["Valuation/Exchange Differences, etc."] - item["Non-Controlling Interests"]
            other_equity.append(temp)

        df_jasa_equity.loc["Other Equity, Total"] = other_equity

        df_jasa_equity = df_jasa_equity.reindex(index=["Total Liabilities",
                                                        "Common Stock",
                                                        "Total, Additional Paid-In",
                                                        "Retained Earnings",
                                                        "Treasury Stock-Common",
                                                        "Valuation/Exchange Differences, etc.",
                                                        "Other Equity, Total",
                                                        "Non-Controlling Interests",
                                                        "Net Assets",
                                                        "Total Liabilities & Net Assets",
                                                        "Total Shareholder's Equity"])

        return df_jasa_equity

    # 収入(revenue)
    def JASA_revenue(df_info, df_key_list, filtered_df):
        jasa_revenue_code = ["Total Revenue 1",
                            "Total Revenue 2",
                            "Total Revenue 3",
                            "Total Revenue 4",
                            "Total Revenue 5",
                            "Cost of Revenue, Total 1",
                            "Cost of Revenue, Total 2",
                            "Gross Profit",
                            "Gross Profit 3"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        join_df = join_df.fillna(0.1)
        df_jasa_revenue = join_df[join_df["JASA"].isin(jasa_revenue_code)]
        df_jasa_revenue = df_jasa_revenue.drop("要素ID", axis = 1)
        df_jasa_revenue.iloc[:, 1:] = df_jasa_revenue.iloc[:, 1:].astype(np.float64)
        df_jasa_revenue = df_jasa_revenue.groupby(["JASA"]).sum()

        revenue_values = []
        for column_name, item in df_jasa_revenue.items():
            if (column_name == "9706" or column_name == "6183" or column_name == "6028" or 
        column_name == "8572" or column_name == "8515" or column_name == "3563" or 
        column_name == "9069" or column_name == "9301" or column_name == "9324" or 
        column_name == "9143" or column_name == "9733" or column_name == "8905" or 
        column_name == "8801" or column_name == "3289" or column_name == "3231" or 
        column_name == "8830" or column_name == "8802" or column_name == "8804" or 
        column_name == "8609" or column_name == "8616" or column_name == "7453" or 
        column_name == "9304" or column_name == "9302" or column_name == "9502" or 
        column_name == "9506" or column_name == "9513" or column_name == "9508" or 
        column_name == "9504" or column_name == "3003"):
                temp = item["Total Revenue 3"]
            elif (column_name == "9602" or column_name == "9206"):
                temp = item["Total Revenue 4"]
            elif (column_name == "6532" or column_name == "8252" or column_name == "6366" or column_name == "6330"):
                temp = item["Total Revenue 2"]
            else:
                temp = item["Total Revenue 1"]
            revenue_values.append(temp)

        df_jasa_revenue.loc["Total Revenue True"] = revenue_values
            
        cost_of_revenue_values = []
        for column_name, item in df_jasa_revenue.items():
            if (column_name == "9602" or column_name == "9069" or column_name == "9301" or 
        column_name == "9324" or column_name == "9143" or column_name == "9733" or 
        column_name == "8905" or column_name == "8801" or column_name == "3289" or 
        column_name == "3231" or column_name == "8830" or column_name == "8802" or 
        column_name == "8804" or column_name == "7453" or column_name == "9304" or 
        column_name == "3003" or column_name == "6366" or column_name == "6330"):
                temp = item["Cost of Revenue, Total 2"]
            else:
                temp = item["Cost of Revenue, Total 1"]
            cost_of_revenue_values.append(temp)

        df_jasa_revenue.loc["Cost of Revenue, Total True"] = cost_of_revenue_values 


        gross_values = []
        for column_name, item in df_jasa_revenue.items():
            if (column_name == "9706" or column_name == "9069" or column_name == "9301" or 
        column_name == "9324" or column_name == "9143" or column_name == "9733" or 
        column_name == "9206" or column_name == "8905" or column_name == "3289" or 
        column_name == "3231" or column_name == "8802" or column_name == "8804" or 
        column_name == "7453" or column_name == "9304" or column_name == "9302" or 
        column_name == "3003" or column_name == "6366" or column_name == "6330"):
                temp = item["Gross Profit 3"]
            else:
                temp = item["Gross Profit"]
            gross_values.append(temp)

        df_jasa_revenue.loc["Gross Profit True"] = gross_values 


        df_jasa_revenue = df_jasa_revenue.reindex(index = ["Total Revenue True",
                                                        "Total Revenue 1",
                                                        "Total Revenue 2",
                                                        "Total Revenue 3",
                                                        "Total Revenue 4",
                                                        "Total Revenue 5",
                                                        "Cost of Revenue, Total True",
                                                    "Cost of Revenue, Total 1",
                                                    "Cost of Revenue, Total 2",
                                                    "Gross Profit True",
                                                    "Gross Profit",
                                                    "Gross Profit 3"])

        return df_jasa_revenue

    # その他PL
    def JASA_pl(df_info, df_key_list, filtered_df):
        jasa_pl_code = ["Total Operating Expenses",
                        "Selling/General/Admin. Expenses, Total",
                        "Personnel Expenses",
                        "Personnel Expenses a",
                        "Research & Development",
                        "Advertising Expenses",
                        "Outsourcing Cost",
                        "Rents",
                        "Depreciation / Amortization",
                        "Depreciation / Amortization a",
                        "Logistics Costs 1",
                        "Logistics Costs 2",
                        "Operating Income 1",
                        "Operating Income 2",
                        "Non-Operating Income",
                        "Interest and Dividend Income PL",
                        "Interest and Dividend Income PL 2",
                        "Interest and Dividend Income PL 3",
                        "Investment Gains on Equity Method",
                        "Gains From Foreign Exchange",
                        "Rental Income",
                        "Gains From Sale of Assets",
                        "Subsidies",
                        "Non-Operating Expenses",
                        "Interest Expenses PL",
                        "Interest Expenses PL 2",
                        "Loss From Sale of Assets",
                        "Investment Loss on Equity Method",
                        "Rental Expenses",
                        "Loss From Foreign Exchange",
                        "Transaction Fees",
                        "Ordinary Profit",
                        "Extraordinary Income",
                        "Extraordinary Loss",
                        "Net Income Before Taxes",
                        "Provision for Income Taxes",
                        "Net Income"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") #ここを消した場合、別記事業codeが入って、error(float + str)が出るはず。でもここが問題か？
        join_df = join_df.fillna(0)
        df_jasa_pl = join_df[join_df["JASA"].isin(jasa_pl_code)]
        df_jasa_pl = df_jasa_pl.drop_duplicates(subset="要素ID")
        df_jasa_pl = df_jasa_pl.drop("要素ID", axis = 1)
        df_jasa_pl.iloc[:, 1:] = df_jasa_pl.iloc[:, 1:].astype(np.float64)
        df_jasa_pl = df_jasa_pl.groupby(["JASA"]).sum()
        filtered_df.iloc[:, 1:] = filtered_df.iloc[:, 1:].astype(np.float64, errors="ignore")

        interest_pl_values = []
        for column_name, item in df_jasa_pl.items():
            if (item["Interest and Dividend Income PL"] <= -50 or item["Interest and Dividend Income PL"] >= 50) and item["Interest and Dividend Income PL"] < item["Non-Operating Income"]:
                temp = item["Interest and Dividend Income PL"]
            elif (item["Interest and Dividend Income PL 2"] <= -50 or item["Interest and Dividend Income PL 2"] >= 50) and item["Interest and Dividend Income PL 2"] < item["Non-Operating Income"]:
                temp = item["Interest and Dividend Income PL 2"]
            else:
                temp = item["Interest and Dividend Income PL 3"]
            interest_pl_values.append(temp)

        df_jasa_pl.loc["Interest and Dividend Income PL True"] = interest_pl_values

        interest_pl_losses = []
        for column_name, item in df_jasa_pl.items():
            if item["Interest Expenses PL"] <= -50 or item["Interest Expenses PL"] >= 50:
                temp = item["Interest Expenses PL"]
            else:
                temp = item["Interest Expenses PL 2"]
            interest_pl_losses.append(temp)

        df_jasa_pl.loc["Interest Expenses PL True"] = interest_pl_losses

        research_values = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Research & Development"]
            try:
                # ここで何らかの処理を行う
                if column_name == str(2229):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E25303-000:ResearchAndDevelopmentExpensesGeneralAndAdministrativeExpenses", str(column_name)])
                elif column_name == str(9783):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04939-000:ResearchAndDevelopmentCostsGeneralAndAdministrativeExpenses", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04939-000:ResearchAndDevelopmentCostsManufacturingCostForCurrentPeriod", str(column_name)])
            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            research_values.append(temp)

        df_jasa_pl.loc["Research & Development"] = research_values

        outsourcing_cost = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Outsourcing Cost"]
            try:
                # ここで何らかの処理を行う
                if column_name == str(9470):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00707-000:WorkConsignmentExpenses", str(column_name)])
                elif column_name == str(9202):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04273-000:OutsourcingFeeSGA", str(column_name)])
                elif column_name == str(9532):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04520-000:ConsignmentWorkExpensesSGA", str(column_name)])
                elif column_name == str(9513):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04510-000:ConsignmentCostSGA", str(column_name)])
            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            outsourcing_cost.append(temp)

        df_jasa_pl.loc["Outsourcing Cost"] = outsourcing_cost

        advertising_cost = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Advertising Expenses"]
            try:
                # ここで何らかの処理を行う
                if column_name == str(2613):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00434-000:AdvertisementSGA", str(column_name)])
                elif column_name == str(2267):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00406-000:SalesPromotionExpensesSGA", str(column_name)])
                elif column_name == str(3382):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03462-000:AdvertisingAndDecorationExpenses", str(column_name)])
                elif column_name == str(4676):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04462-000:AdvertisementSellingExpenses", str(column_name)])
            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            advertising_cost.append(temp)

        df_jasa_pl.loc["Advertising Expenses"] = advertising_cost

        rent = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Rents"]
            try:
                if column_name == str(9302):
                    temp -= float(filtered_df.loc["jppfs_cor:RentExpensesSGA", str(column_name)])
                elif column_name == str(7816):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:Rents", str(column_name)])
            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            rent.append(temp)

        df_jasa_pl.loc["Rents"] = rent

        depreciation_values = []
        for column_name, item in df_jasa_pl.items():
            if item["Depreciation / Amortization"] <= -50 or item["Depreciation / Amortization"] >= 50:
                temp = item["Depreciation / Amortization"] + item["Depreciation / Amortization a"]
                try:
                    if column_name == str(5902):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:AmortizationOfGoodwill", str(column_name)])
                    elif column_name == str(6532):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E32549-000:DepreciationAndAmortizationSGA", str(column_name)])
                except KeyError:
                    print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            else:
                temp = item["Depreciation / Amortization a"]
                try:
                    if column_name == str(5902):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:AmortizationOfGoodwill", str(column_name)])
                    elif column_name == str(6532):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E32549-000:DepreciationAndAmortizationSGA", str(column_name)])

                except KeyError:
                    print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            depreciation_values.append(temp)

        df_jasa_pl.loc["Depreciation / Amortization True"] = depreciation_values 
        
        personnel_values = []
        for column_name, item in df_jasa_pl.items():
            if item["Personnel Expenses"] <= -50 or item["Personnel Expenses"] >= 50:
                temp = item["Personnel Expenses"] + item["Personnel Expenses a"]
                try:
                    if column_name == str(7972):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02371-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E02371-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                    elif column_name == str(2168):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E05729-000:EmployeePayrollsAndCompensationAndOtherSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E05729-000:EmployeePayrollsAndCompensationAndOtherSGA", str(column_name)])
                    elif column_name == str(9278):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E34102-000:SalariesOfPartTimeEmployeesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E34102-000:SalariesOfPartTimeEmployeesSGA", str(column_name)])
                    elif column_name == str(2229):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E25303-000:PayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E25303-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(9861):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03153-000:CostForPartTimersSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E03153-000:CostForPartTimersSGA", str(column_name)])
                    elif column_name == str(7581):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03305-000:EmployeesPayrollsAndBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E03305-000:EmployeesPayrollsAndBonusesSGA", str(column_name)])
                    elif column_name == str(4934):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E36046-000:RetirementBenefitExpense", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E36046-000:RetirementBenefitExpense", str(column_name)])
                    elif column_name == str(4031):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00789-000:PayrollsAllowancesAndBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00789-000:PayrollsAllowancesAndBonusesSGA", str(column_name)])
                    elif column_name == str(6753):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01773-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01773-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                    elif column_name == str(6367):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01570-000:DirectorsAndEmployeePayrollsAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01570-000:DirectorsAndEmployeePayrollsAndAllowancesSGA", str(column_name)])
                    elif column_name == str(9301):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                    elif column_name == str(5929):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01385-000:ProvisionForEmployeeCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01385-000:ProvisionForEmployeeCompensationSGA", str(column_name)])
                    elif column_name == str(3407):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00877-000:SalariesAndBenefitsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                    elif column_name == str(3405):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00876-000:PayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00876-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(7269):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02167-000:PayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E02167-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(8905):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E04002-000:ProvisionForDirectorsRemunerationBasedOnPerformanceSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E04002-000:ProvisionForDirectorsRemunerationBasedOnPerformanceSGA", str(column_name)])
                    elif column_name == str(2871):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00446-000:DirectorsCompensationEmployeesSalariesBonusesAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00446-000:DirectorsCompensationEmployeesSalariesBonusesAndAllowancesSGA", str(column_name)])
                    elif column_name == str(2607):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00431-000:EmployeePayrollsAndAllowances", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00431-000:EmployeePayrollsAndAllowances", str(column_name)])
                    elif column_name == str(2004):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00348-000:EmployeeSalariesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00348-000:EmployeeSalariesSGA", str(column_name)])
                    elif column_name == str(6332):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(6508):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01744-000:BonusesAndProvisionForBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01744-000:BonusesAndProvisionForBonusesSGA", str(column_name)])
                    elif column_name == str(6368):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01571-000:EmployeePayrollsAllowancesAndCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01571-000:EmployeePayrollsAllowancesAndCompensationSGA", str(column_name)])
                    elif column_name == str(1301):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:SalesStaffPayrollsAndAllowancesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:OfficeStaffPayrollsAndAllowancesGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(5706):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00024-000:BonusAndRetirementPaymentsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00024-000:BonusAndRetirementPaymentsSGA", str(column_name)])
                    elif column_name == str(9304):
                        temp -= float(filtered_df.loc["jppfs_cor:PersonalExpensesSGA", str(column_name)])
                        #print(filtered_df.loc["jppfs_cor:PersonalExpensesSGA", str(column_name)])
                    elif column_name == str(9302):
                        temp -= float(filtered_df.loc["jppfs_cor:SalariesAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jppfs_cor:SalariesAndAllowancesSGA", str(column_name)])
                    elif column_name == str(2281):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00335-000:PayrollsAndOtherAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00335-000:PayrollsAndOtherAllowancesSGA", str(column_name)])
                    elif column_name == str(2264):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00331-000:EmployeesSalariesAndBonusesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00331-000:EmployeesSalariesAndBonusesGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(2270):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E23202-000:SalariesGeneralAndAdministrativeExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E23202-000:OtherGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E23202-000:SalariesGeneralAndAdministrativeExpenses", str(column_name)])
                    elif column_name == str(5333):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01137-000:PayrollsAndCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01137-000:PayrollsAndCompensationSGA", str(column_name)])
                    elif column_name == str(3382):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03462-000:SalariesAndWages", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E03462-000:SalariesAndWages", str(column_name)])
                    elif column_name == str(5902):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:ProvisionForManagementBoardIncentivePlanTrust", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:ProvisionForEmployeeStockOwnershipPlanTrust", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(6861):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01967-000:DirectorsCompensationAndEmployeeSalariesAllowancesAndCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01967-000:DirectorsCompensationAndEmployeeSalariesAllowancesAndCompensationSGA", str(column_name)])
                    elif column_name == str(7816):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:ProvisionForDirectorSPerformanceLinkedIncentiveCompensationSGA", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:ProvisionForEmployeeSPerformanceLinkedIncentiveCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(5233):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01130-000:PersonnelExpenditureSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01130-000:PersonnelExpenditureSGA", str(column_name)])
                    elif column_name == str(5232):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01127-000:PayrollsAndBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01127-000:PayrollsAndBonusesSGA", str(column_name)])
                    elif column_name == str(4043):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00768-000:SalariesBonusesAndAllowancesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00768-000:SalariesBonusesAndAllowancesGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(3191):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E30501-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(9206):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E26084-000:PayrollsAndAllowancesSGA", str(column_name)])

                    
                except KeyError:
                    print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")


            else:
                temp = item["Personnel Expenses a"]
                try:
                    if column_name == str(7972):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02371-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E02371-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                    elif column_name == str(2168):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E05729-000:EmployeePayrollsAndCompensationAndOtherSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E05729-000:EmployeePayrollsAndCompensationAndOtherSGA", str(column_name)])
                    elif column_name == str(9278):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E34102-000:SalariesOfPartTimeEmployeesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E34102-000:SalariesOfPartTimeEmployeesSGA", str(column_name)])
                    elif column_name == str(2229):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E25303-000:PayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E25303-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(9861):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03153-000:CostForPartTimersSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E03153-000:CostForPartTimersSGA", str(column_name)])
                    elif column_name == str(7581):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03305-000:EmployeesPayrollsAndBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E03305-000:EmployeesPayrollsAndBonusesSGA", str(column_name)])
                    elif column_name == str(4934):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E36046-000:RetirementBenefitExpense", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E36046-000:RetirementBenefitExpense", str(column_name)])
                    elif column_name == str(4031):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00789-000:PayrollsAllowancesAndBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00789-000:PayrollsAllowancesAndBonusesSGA", str(column_name)])
                    elif column_name == str(6753):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01773-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01773-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                    elif column_name == str(6367):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01570-000:DirectorsAndEmployeePayrollsAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01570-000:DirectorsAndEmployeePayrollsAndAllowancesSGA", str(column_name)])
                    elif column_name == str(9301):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                    elif column_name == str(5929):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01385-000:ProvisionForEmployeeCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01385-000:ProvisionForEmployeeCompensationSGA", str(column_name)])
                    elif column_name == str(3407):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00877-000:SalariesAndBenefitsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                    elif column_name == str(3405):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00876-000:PayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00876-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(7269):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02167-000:PayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E02167-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(8905):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E04002-000:ProvisionForDirectorsRemunerationBasedOnPerformanceSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E04002-000:ProvisionForDirectorsRemunerationBasedOnPerformanceSGA", str(column_name)])
                    elif column_name == str(2871):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00446-000:DirectorsCompensationEmployeesSalariesBonusesAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00446-000:DirectorsCompensationEmployeesSalariesBonusesAndAllowancesSGA", str(column_name)])
                    elif column_name == str(2607):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00431-000:EmployeePayrollsAndAllowances", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00431-000:EmployeePayrollsAndAllowances", str(column_name)])
                    elif column_name == str(2004):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00348-000:EmployeeSalariesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00348-000:EmployeeSalariesSGA", str(column_name)])
                    elif column_name == str(6332):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(6508):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01744-000:BonusesAndProvisionForBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01744-000:BonusesAndProvisionForBonusesSGA", str(column_name)])
                    elif column_name == str(6368):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01571-000:EmployeePayrollsAllowancesAndCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01571-000:EmployeePayrollsAllowancesAndCompensationSGA", str(column_name)])
                    elif column_name == str(1301):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:SalesStaffPayrollsAndAllowancesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:OfficeStaffPayrollsAndAllowancesGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(5706):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00024-000:BonusAndRetirementPaymentsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00024-000:BonusAndRetirementPaymentsSGA", str(column_name)])
                    elif column_name == str(9304):
                        temp -= float(filtered_df.loc["jppfs_cor:PersonalExpensesSGA", str(column_name)])
                        #print(filtered_df.loc["jppfs_cor:PersonalExpensesSGA", str(column_name)])
                    elif column_name == str(9302):
                        temp -= float(filtered_df.loc["jppfs_cor:SalariesAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jppfs_cor:SalariesAndAllowancesSGA", str(column_name)])
                    elif column_name == str(2281):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00335-000:PayrollsAndOtherAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00335-000:PayrollsAndOtherAllowancesSGA", str(column_name)])
                    elif column_name == str(2264):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00331-000:EmployeesSalariesAndBonusesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00331-000:EmployeesSalariesAndBonusesGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(2270):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E23202-000:SalariesGeneralAndAdministrativeExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E23202-000:OtherGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E23202-000:SalariesGeneralAndAdministrativeExpenses", str(column_name)])
                    elif column_name == str(5333):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01137-000:PayrollsAndCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01137-000:PayrollsAndCompensationSGA", str(column_name)])
                    elif column_name == str(3382):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03462-000:SalariesAndWages", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E03462-000:SalariesAndWages", str(column_name)])
                    elif column_name == str(5902):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:ProvisionForManagementBoardIncentivePlanTrust", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:ProvisionForEmployeeStockOwnershipPlanTrust", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(6861):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01967-000:DirectorsCompensationAndEmployeeSalariesAllowancesAndCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01967-000:DirectorsCompensationAndEmployeeSalariesAllowancesAndCompensationSGA", str(column_name)])
                    elif column_name == str(7816):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:ProvisionForDirectorSPerformanceLinkedIncentiveCompensationSGA", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:ProvisionForEmployeeSPerformanceLinkedIncentiveCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(5233):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01130-000:PersonnelExpenditureSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01130-000:PersonnelExpenditureSGA", str(column_name)])
                    elif column_name == str(5232):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01127-000:PayrollsAndBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01127-000:PayrollsAndBonusesSGA", str(column_name)])
                    elif column_name == str(4043):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00768-000:SalariesBonusesAndAllowancesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00768-000:SalariesBonusesAndAllowancesGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(3191):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E30501-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(9206):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E26084-000:PayrollsAndAllowancesSGA", str(column_name)])

                except KeyError:
                    print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            personnel_values.append(temp)

        df_jasa_pl.loc["Personnel Expenses True"] = personnel_values


        logistics_costs = []
        for column_name, item in df_jasa_pl.items():
            if item["Logistics Costs 1"] >= 50:
                temp = item["Logistics Costs 1"]
                try:
                    if column_name == str(8086):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02688-000:TransportationExpenses2SGA", str(column_name)])
                    elif column_name == str(6367):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01570-000:ProductFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(8173):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03052-000:DistributionExpenditureSGA", str(column_name)])
                    elif column_name == str(2211):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00374-000:FreightageAndStorageFeeSGA", str(column_name)])
                    elif column_name == str(9069):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E04179-000:UnderPaymentFareSGA", str(column_name)])
                    elif column_name == str(5332):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01138-000:DistributionExpenses", str(column_name)])
                    elif column_name == str(5932):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E26831-000:PackingMaterialsAndFreightageExpenditureSGA", str(column_name)])
                    elif column_name == str(3405):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00876-000:FreightageAndWarehouseExpensesSGA", str(column_name)])
                    elif column_name == str(7269):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02167-000:ShippingFee", str(column_name)])
                    elif column_name == str(2602):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00428-000:ProductsFreightageAndStorageFeeSGA", str(column_name)])
                    elif column_name == str(2613):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00434-000:ProductsFreightageFeeSGA", str(column_name)])
                    elif column_name == str(2004):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00348-000:ShippingAndDeliveryExpensesSGA", str(column_name)])
                    elif column_name == str(1301):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:ShippingAndDeliveryExpensesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:StorageExpensesSGA", str(column_name)])
                    elif column_name == str(7538):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02818-000:ShippingBountyAndCompletedDeliveryBountySGA", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02818-000:FreightExpensesSGA", str(column_name)])
                    elif column_name == str(8030):
                        temp += float(filtered_df.loc["jppfs_cor:SalesCommissionSGA", str(column_name)])
                    elif column_name == str(7453):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03248-000:DistributionExpensesAndTransportationCostsSGA", str(column_name)])
                    elif column_name == str(3315):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00030-000:OceanFreightSGA", str(column_name)])
                    elif column_name == str(3106):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00528-000:FreightOutExpenseWarehousingCostAndPackingExpenseSGA", str(column_name)])
                    elif column_name == str(2296):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E32069-000:ShippingAndDeliveryExpensesSGA", str(column_name)])
                    elif column_name == str(2281):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00335-000:PackingAndFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(1383):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E25969-000:PackingAndFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(5698):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E27868-000:TransportationSundryExpensesSGA", str(column_name)])
                    elif column_name == str(3864):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00644-000:PackingAndFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(3865):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00645-000:TransportationExpenditureSGA", str(column_name)])
                    elif column_name == str(5233):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01130-000:SalesFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(7513):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03270-000:TransportationCostsSGA", str(column_name)])
                    elif column_name == str(1925):
                        temp += float(filtered_df.loc["jppfs_cor:SalesCommissionSGA", str(column_name)])
                    elif column_name == str(3864):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00644-000:PackingAndFreightageExpensesSGA", str(column_name)])
 
                except KeyError:
                    print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")


            else:
                temp = item["Logistics Costs 2"]
                try:
                    if column_name == str(8086):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02688-000:TransportationExpenses2SGA", str(column_name)])
                    elif column_name == str(6367):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01570-000:ProductFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(8173):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03052-000:DistributionExpenditureSGA", str(column_name)])
                    elif column_name == str(2211):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00374-000:FreightageAndStorageFeeSGA", str(column_name)])
                    elif column_name == str(9069):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E04179-000:UnderPaymentFareSGA", str(column_name)])
                    elif column_name == str(5332):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01138-000:DistributionExpenses", str(column_name)])
                    elif column_name == str(5932):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E26831-000:PackingMaterialsAndFreightageExpenditureSGA", str(column_name)])
                    elif column_name == str(3405):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00876-000:FreightageAndWarehouseExpensesSGA", str(column_name)])
                    elif column_name == str(7269):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02167-000:ShippingFee", str(column_name)])
                    elif column_name == str(2602):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00428-000:ProductsFreightageAndStorageFeeSGA", str(column_name)])
                    elif column_name == str(2613):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00434-000:ProductsFreightageFeeSGA", str(column_name)])
                    elif column_name == str(2004):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00348-000:ShippingAndDeliveryExpensesSGA", str(column_name)])
                    elif column_name == str(1301):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:ShippingAndDeliveryExpensesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:StorageExpensesSGA", str(column_name)])
                    elif column_name == str(7538):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02818-000:ShippingBountyAndCompletedDeliveryBountySGA", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02818-000:FreightExpensesSGA", str(column_name)])
                    elif column_name == str(8030):
                        temp += float(filtered_df.loc["jppfs_cor:SalesCommissionSGA", str(column_name)])
                    elif column_name == str(7453):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03248-000:DistributionExpensesAndTransportationCostsSGA", str(column_name)])
                    elif column_name == str(3315):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00030-000:OceanFreightSGA", str(column_name)])
                    elif column_name == str(3106):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00528-000:FreightOutExpenseWarehousingCostAndPackingExpenseSGA", str(column_name)])
                    elif column_name == str(2296):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E32069-000:ShippingAndDeliveryExpensesSGA", str(column_name)])
                    elif column_name == str(2281):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00335-000:PackingAndFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(1383):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E25969-000:PackingAndFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(5698):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E27868-000:TransportationSundryExpensesSGA", str(column_name)])
                    elif column_name == str(3864):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00644-000:PackingAndFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(3865):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00645-000:TransportationExpenditureSGA", str(column_name)])
                    elif column_name == str(5233):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01130-000:SalesFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(7513):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03270-000:TransportationCostsSGA", str(column_name)])
                    elif column_name == str(1925):
                        temp += float(filtered_df.loc["jppfs_cor:SalesCommissionSGA", str(column_name)])
                    elif column_name == str(3864):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00644-000:PackingAndFreightageExpensesSGA", str(column_name)])

                except KeyError:
                    print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            logistics_costs.append(temp)

        df_jasa_pl.loc["Logistics Costs True"] = logistics_costs

        rental_expenses = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Rental Expenses"]
            try:
                if column_name == str(1967):
                    temp += float(filtered_df.loc["jppfs_cor:RentExpensesNOE", str(column_name)])
            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            rental_expenses.append(temp)

        df_jasa_pl.loc["Rental Expenses"] = rental_expenses


        operating_values = []
        for column_name, item in df_jasa_pl.items():
            if item["Operating Income 1"] <= -50 or item["Operating Income 1"] >= 50:
                temp = item["Operating Income 1"]
            else :
                temp = item["Operating Income 2"]
            operating_values.append(temp)  

        df_jasa_pl.loc["Operating Income True"] = operating_values  

        other_sga = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Selling/General/Admin. Expenses, Total"] - item["Logistics Costs True"] - item["Personnel Expenses True"] - item["Advertising Expenses"] - item["Outsourcing Cost"] - item["Rents"] - item["Research & Development"] - item["Depreciation / Amortization True"]
            other_sga.append(temp)
        df_jasa_pl.loc["Other Selling/General/Admin. Expenses"] = other_sga

        other_noi = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Non-Operating Income"] - item["Interest and Dividend Income PL True"] - item["Investment Gains on Equity Method"] - item["Gains From Foreign Exchange"] - item["Rental Income"] - item["Subsidies"]
            other_noi.append(temp)
        df_jasa_pl.loc["Other Non-Operating Income"] = other_noi

        other_noe = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Non-Operating Expenses"] - item["Interest Expenses PL True"] - item["Investment Loss on Equity Method"] - item["Loss From Foreign Exchange"] - item["Rental Expenses"] - item["Transaction Fees"]
            other_noe.append(temp)
        df_jasa_pl.loc["Other Non-Operating Expenses"] = other_noe

        df_jasa_pl = df_jasa_pl.reindex(index=["Total Operating Expenses",
                                                    "Selling/General/Admin. Expenses, Total",
                                                    "Logistics Costs True",
                                                    "Personnel Expenses True",
                                                    "Advertising Expenses",
                                                    "Outsourcing Cost",
                                                    "Rents",
                                                    "Research & Development",
                                                    "Depreciation / Amortization True",
                                                    "Other Selling/General/Admin. Expenses",
                                                    "Operating Income True",
                                                    "Non-Operating Income",
                                                    "Interest and Dividend Income PL True",
                                                    "Investment Gains on Equity Method",
                                                    "Gains From Foreign Exchange",
                                                    "Rental Income",
                                                    "Subsidies",
                                                    "Other Non-Operating Income",
                                                    "Non-Operating Expenses",
                                                    "Interest Expenses PL True",
                                                    "Investment Loss on Equity Method",
                                                    "Loss From Foreign Exchange",
                                                    "Rental Expenses",
                                                    "Transaction Fees",
                                                    "Other Non-Operating Expenses",
                                                    "Ordinary Profit",
                                                    "Extraordinary Income",
                                                    "Extraordinary Loss",
                                                    "Net Income Before Taxes",
                                                    "Provision for Income Taxes",
                                                    "Net Income"])

        return df_jasa_pl

    # 営業CF
    def JASA_opcf(df_info, df_key_list, filtered_df):
        jasa_opcf_code = ["Net Income Before Taxes",
                        "Depreciation/Depletion",
                        "Amortization of Goodwill",
                        "Depreciation and Amortization on Other",
                        "Increase (Decrease) in Provision",
                        "Interest and Dividend Income CF",
                        "Interest Expenses CF",
                        "Share of Loss (Profit) of Entities Accounted for Using Equity Method",
                        "Other Loss (Gain)",
                        "Decrease (Increase) in Trade Receivables",
                        "Decrease (Increase) in Inventories",
                        "Increase (Decrease) in Advances Received",
                        "Decrease (Increase) in Inventories",
                        "Increase (Decrease) in Advances Received",
                        "Decrease (Increase) in Prepaid Expenses",
                        "Increase (Decrease) in Trade Payables",
                        "Increase (Decrease) in Retirement Benefit Liabilities",
                        "Increase (Decrease) in Accounts Payable - Other, and Accrued Expenses",
                        "Subtotal",
                        "Interest and Dividends Received",
                        "Interest Paid",
                        "Proceeds from Insurance Income",
                        "Compensation Paid for Damage",
                        "Cash Taxes Paid",
                        "Cash from Operating Activities"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") 
        join_df = join_df.fillna(0.1)
        df_jasa_opcf = join_df[join_df["JASA"].isin(jasa_opcf_code)]
        df_jasa_opcf = df_jasa_opcf.drop("要素ID", axis = 1)
        df_jasa_opcf.iloc[:, 1:] = df_jasa_opcf.iloc[:, 1:].astype(np.float64)
        df_jasa_opcf = df_jasa_opcf.groupby(["JASA"]).sum()

        other_opcf = []
        for column_name, item in df_jasa_opcf.items():
            temp = item["Subtotal"]

            items_to_subtract = ["Net Income Before Taxes",
                "Depreciation/Depletion",
                "Amortization of Goodwill",
                "Depreciation and Amortization on Other",
                "Increase (Decrease) in Provision",
                "Interest and Dividend Income CF",
                "Interest Expenses CF",
                "Share of Loss (Profit) of Entities Accounted for Using Equity Method",
                "Other Loss (Gain)",
                "Decrease (Increase) in Trade Receivables",
                "Decrease (Increase) in Inventories",
                "Increase (Decrease) in Advances Received",
                "Decrease (Increase) in Prepaid Expenses",
                "Increase (Decrease) in Trade Payables",
                "Increase (Decrease) in Retirement Benefit Liabilities",
                "Increase (Decrease) in Accounts Payable - Other, and Accrued Expenses"
            ]
            for key in items_to_subtract:
                temp -= item[key]
            other_opcf.append(temp)

        df_jasa_opcf.loc["Other, Net"] = other_opcf


        other_pr = []
        for column_name, item in df_jasa_opcf.items():
            temp = item["Cash from Operating Activities"]
            # 引く項目のリスト
            items_to_subtract = [
                "Subtotal",
                "Interest and Dividends Received",
                "Interest Paid",
                "Proceeds from Insurance Income",
                "Compensation Paid for Damage",
                "Cash Taxes Paid"
            ]
            for key in items_to_subtract:
                temp -= item[key]
            other_pr.append(temp)
        
        df_jasa_opcf.loc["Other Cash from Operating Activities"] = other_pr

        df_jasa_opcf = df_jasa_opcf.rename(index={"Net Income Before Taxes":"Net Income/Starting Line"})

        df_jasa_opcf = df_jasa_opcf.reindex(index=["Net Income/Starting Line",
                                                  "Depreciation/Depletion",
                                                    "Amortization of Goodwill",
                                                    "Depreciation and Amortization on Other",
                                                    "Increase (Decrease) in Provision",
                                                    "Interest and Dividend Income CF",
                                                    "Interest Expenses CF",
                                                    "Share of Loss (Profit) of Entities Accounted for Using Equity Method",
                                                    "Other Loss (Gain)",
                                                    "Decrease (Increase) in Trade Receivables",
                                                    "Decrease (Increase) in Inventories",
                                                    "Increase (Decrease) in Advances Received",
                                                    "Decrease (Increase) in Inventories",
                                                    "Increase (Decrease) in Advances Received",
                                                    "Decrease (Increase) in Prepaid Expenses",
                                                    "Increase (Decrease) in Trade Payables",
                                                    "Increase (Decrease) in Retirement Benefit Liabilities",
                                                    "Increase (Decrease) in Accounts Payable - Other, and Accrued Expenses",
                                                    "Other, Net",
                                                    "Subtotal",
                                                    "Interest and Dividends Received",
                                                    "Interest Paid",
                                                    "Proceeds from Insurance Income",
                                                    "Compensation Paid for Damage",
                                                    "Cash Taxes Paid",
                                                    "Other Cash from Operating Activities",
                                                    "Cash from Operating Activities"])

        return df_jasa_opcf

    # 営業CFの直接法ver.
    def JASA_opcf2(df_info, df_key_list, filtered_df):
        jasa_opcf2_code = ["Receipts from Operating Revenue",
                           "Payments for Raw Materials and Goods",
                           "Payments of Personnel Expenses",
                           "Subtotal",
                           "Interest and Dividends Received",
                            "Interest Paid",
                            "Proceeds from Insurance Income",
                            "Compensation Paid for Damage",
                            "Cash Taxes Paid",
                            "Cash from Operating Activities"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") 
        join_df = join_df.fillna(0.1)
        df_jasa_opcf2 = join_df[join_df["JASA"].isin(jasa_opcf2_code)]
        df_jasa_opcf2 = df_jasa_opcf2.drop("要素ID", axis = 1)
        df_jasa_opcf2.iloc[:, 1:] = df_jasa_opcf2.iloc[:, 1:].astype(np.float64)
        df_jasa_opcf2 = df_jasa_opcf2.groupby(["JASA"]).sum()
        
        other_opcf2 = []
        for column_name, item in df_jasa_opcf2.items():
            temp = item["Subtotal"]
            # 引く項目のリスト
            items_to_subtract = [
                "Receipts from Operating Revenue",
                "Payments for Raw Materials and Goods",
                "Payments of Personnel Expenses"
            ]
            for key in items_to_subtract:
                temp -= item[key]
            other_opcf2.append(temp)
        
        df_jasa_opcf2.loc["Other, Net"] = other_opcf2

        other_pr2 = []
        for column_name, item in df_jasa_opcf2.items():
            temp = item["Cash from Operating Activities"]
            # 引く項目のリスト
            items_to_subtract = [
                "Subtotal",
                "Interest and Dividends Received",
                "Interest Paid",
                "Proceeds from Insurance Income",
                "Compensation Paid for Damage",
                "Cash Taxes Paid"
            ]
            for key in items_to_subtract:
                temp -= item[key]
            other_pr2.append(temp)

        df_jasa_opcf2.loc["Other Cash from Operating Activities"] = other_pr2

        return df_jasa_opcf2

    # 投資CF
    def JASA_incf(df_info, df_key_list, filtered_df):
        jasa_incf_code = ["Purchase of Securities",
                          "Proceeds from Sale of Securities",
                          "Purchase of Property, Plant and Equipment",
                          "Proceeds from Sale of Property, Plant and Equipment",
                          "Purchase of Intangible Assets",
                          "Proceeds from Sale of Intangible Assets",
                          "Purchase of Investment Securities",
                          "Proceeds from Sale of Investment Securities",
                          "Purchase of Shares of Subsidiaries Resulting in Change in Scope of Consolidation",
                          "Proceeds from Sale of Shares of Subsidiaries Resulting in Change in Scope of Consolidation",
                          "Loans Advances",
                          "Proceeds from Collection of Loans Receivable",
                          "Payments into Time Deposits",
                          "Proceeds from Withdrawal of Time Deposits",
                          "Payments of Leasehold and Guarantee Deposits",
                          "Proceeds from Refund of Leasehold and Guarantee Deposits",
                          "Payments for Acquisition of Businesses",
                          "Proceeds from Sale of Businesses",
                          "Cash from Investing Activities"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") 
        join_df = join_df.fillna(0.1)
        df_jasa_incf = join_df[join_df["JASA"].isin(jasa_incf_code)]
        df_jasa_incf = df_jasa_incf.drop("要素ID", axis = 1)
        df_jasa_incf.iloc[:, 1:] = df_jasa_incf.iloc[:, 1:].astype(np.float64)
        df_jasa_incf = df_jasa_incf.groupby(["JASA"]).sum()

        other_incf = []
        for column_name, item in df_jasa_incf.items():
            temp = item["Cash from Investing Activities"]
            # 引く項目のリスト
            items_to_subtract = [
                "Purchase of Securities",
                "Proceeds from Sale of Securities",
                "Purchase of Property, Plant and Equipment",
                "Proceeds from Sale of Property, Plant and Equipment",
                "Purchase of Intangible Assets",
                "Proceeds from Sale of Intangible Assets",
                "Purchase of Investment Securities",
                "Proceeds from Sale of Investment Securities",
                "Purchase of Shares of Subsidiaries Resulting in Change in Scope of Consolidation",
                "Proceeds from Sale of Shares of Subsidiaries Resulting in Change in Scope of Consolidation",
                "Loans Advances",
                "Proceeds from Collection of Loans Receivable",
                "Payments into Time Deposits",
                "Proceeds from Withdrawal of Time Deposits",
                "Payments of Leasehold and Guarantee Deposits",
                "Proceeds from Refund of Leasehold and Guarantee Deposits",
                "Payments for Acquisition of Businesses",
                "Proceeds from Sale of Businesses"]
            for key in items_to_subtract:
                temp -= item[key]
            other_incf.append(temp)
        
        df_jasa_incf.loc["Other Cash from Investing Activities"] = other_incf

        return df_jasa_incf

    # 財務CF
    def JASA_ficf(df_info, df_key_list, filtered_df):
        jasa_ficf_code= [
                          "Proceeds from Short Term Borrowings",
                          "Repayments of Short Term Borrowings",
                          "Proceeds from Long Term Borrowings",
                          "Repayments of Long Term Borrowings",
                          "Proceeds from Issuance of Bonds",
                          "Redemption of Bonds",
                          "Proceeds from Issuance of Shares",
                          "Proceeds from Share Issuance to Non-Controlling Shareholders",
                          "Proceeds from Sale of Shares of Subsidiaries Not Resulting in Change in Scope of Consolidation",
                          "Purchase of Shares of Subsidiaries Not Resulting in Change in Scope of Consolidation",
                          "Purchase of Treasury Shares",
                          "Proceeds from Sale of Treasury Shares",
                          "Repayments of Lease Liabilities",
                          "Dividends Paid",
                          "Dividends Paid to Non-Controlling Interests",
                          "Cash from Financing Activities",
                          "Foreign Exchange Effects",
                          "Net Change in Cash",
                          "Ending Cash Balance"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") 
        join_df = join_df.fillna(0.1)
        df_jasa_ficf = join_df[join_df["JASA"].isin(jasa_ficf_code)]
        df_jasa_ficf = df_jasa_ficf.drop("要素ID", axis = 1)
        df_jasa_ficf.iloc[:, 1:] = df_jasa_ficf.iloc[:, 1:].astype(np.float64)
        df_jasa_ficf = df_jasa_ficf.groupby(["JASA"]).sum()

        other_ficf = []
        for column_name, item in df_jasa_ficf.items():
            temp = item["Cash from Financing Activities"]
            # 引く項目のリスト
            items_to_subtract = [
               "Proceeds from Short Term Borrowings",
                "Repayments of Short Term Borrowings",
                "Proceeds from Long Term Borrowings",
                "Repayments of Long Term Borrowings",
                "Proceeds from Issuance of Bonds",
                "Redemption of Bonds",
                "Proceeds from Share Issuance to Non-Controlling Shareholders",
                "Proceeds from Sale of Shares of Subsidiaries Not Resulting in Change in Scope of Consolidation",
                "Purchase of Shares of Subsidiaries Not Resulting in Change in Scope of Consolidation",
                "Purchase of Treasury Shares",
                "Proceeds from Sale of Treasury Shares",
                "Repayments of Lease Liabilities",
                "Dividends Paid",
                "Dividends Paid to Non-Controlling Interests"]
            
            for key in items_to_subtract:
                temp -= item[key]
            other_ficf.append(temp)
        
        df_jasa_ficf.loc["Other Cash from Financing Activities"] = other_ficf

        return df_jasa_ficf


    def JASA_date(df_info, df_key_list):
        jasa_date_code = ["jpdei_cor:CurrentPeriodEndDateDEI"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") #ここを消した場合、別記事業codeが入って、error(float + str)が出るはず。でもここが問題か？
        join_df = join_df.fillna(0.1)
        df_jasa_date = join_df[join_df["要素ID"].isin(jasa_date_code)]
        df_jasa_date = df_jasa_date.drop_duplicates(subset="要素ID")
        df_jasa_date = df_jasa_date.drop("要素ID", axis = 1)
        #df_jasa_date.iloc[:, 1:] = df_jasa_date.iloc[:, 1:].astype(np.float64)
        #df_jasa_equity = df_jasa_equity.groupby(["JASA"]).sum()

        return df_jasa_date
    
    df_jasa_date = JASA_date(df_info, df_key_list)
    df_jasa_ca = JASA_ca(df_info, df_key_list, filtered_df)
    df_jasa_nca = JASA_nca(df_info, df_key_list, filtered_df)
    df_jasa_cl = JASA_cl(df_info, df_key_list, filtered_df)
    df_jasa_ncl = JASA_ncl(df_info, df_key_list, filtered_df)
    df_jasa_equity = JASA_equity(df_info, df_key_list, filtered_df)
    df_jasa_revenue = JASA_revenue(df_info, df_key_list, filtered_df)
    df_jasa_pl = JASA_pl(df_info, df_key_list, filtered_df)
    df_jasa_opcf = JASA_opcf(df_info, df_key_list, filtered_df)
    df_jasa_opcf2 = JASA_opcf2(df_info, df_key_list, filtered_df)
    df_jasa_incf = JASA_incf(df_info, df_key_list, filtered_df)
    df_jasa_ficf = JASA_ficf(df_info, df_key_list, filtered_df)

    df1 = df_jasa_ca
    df2 = pd.concat([df1, df_jasa_nca])
    df3 = pd.concat([df2, df_jasa_cl])
    df4 = pd.concat([df3, df_jasa_ncl])
    df5 = pd.concat([df4, df_jasa_equity])
    df6 = pd.concat([df5, df_jasa_revenue])
    df7 = pd.concat([df6, df_jasa_pl])
    df8 = pd.concat([df7, df_jasa_opcf])
    df9 = pd.concat([df8, df_jasa_incf])
    df10 = pd.concat([df9, df_jasa_ficf])
    df10 = df10[~df10.index.duplicated(keep='first')]  # 最初に登場するもの以外の重複を削除

    df_jasa = df10.reindex(index=["Total Current Assets", 
                                "Cash and Short Term Investments",
                                "Notes Receivable/Accounts Receivable True",
                                "Total Inventory True",
                                "Prepaid Expenses",
                                "Trading Securities",
                                "Other Current Assets",
                                "Total Non-Current Assets",
                                "Property/Plant/Equipment, Total - Net",
                                "Goodwill, Net",
                                "Other Intangibles, Net",
                                "Long Term Investments",
                                "Leasehold and Guarantee Deposits True",
                                "Deferred Tax Assets",
                                "Other Non-Current Assets",
                                "Total Assets",
                                "Total Current Liabilities",
                                "Notes Payable/Accounts Payable True",
                                "Advances Received True",
                                "Capital Leases",
                                "Short Term Debt",
                                "Current Port. of LT Debt",
                                "Provision CL True",
                                "Other Current Liabilities",
                                "Total Non-Current Liabilities",
                                "Long Term Debt",
                                "Bonds",
                                "Lease Liabilities",
                                "Asset Retirement Obligation",
                                "Retirement Benefit",
                                "Deferred Tax Liabilities",
                                "Provision NCL True",
                                "Other Non-Current Liabilities",
                                "Total Liabilities",
                                "Common Stock",
                                "Total, Additional Paid-In",
                                "Retained Earnings",
                                "Treasury Stock-Common",
                                "Valuation/Exchange Differences, etc.",
                                "Other Equity, Total",
                                "Non-Controlling Interests",
                                "Net Assets",
                                "Total Liabilities & Net Assets",
                                " ",
                                "Total Revenue True",
                                "Cost of Revenue, Total True",
                                "Gross Profit True",
                                "Total Operating Expenses",
                                "Selling/General/Admin. Expenses, Total",
                                "Logistics Costs True",
                                "Personnel Expenses True",
                                "Advertising Expenses",
                                "Outsourcing Cost",
                                "Rents",
                                "Research & Development",
                                "Depreciation / Amortization True",
                                "Other Selling/General/Admin. Expenses",
                                "Operating Income True",
                                "Non-Operating Income",
                                "Interest and Dividend Income PL True",
                                "Investment Gains on Equity Method",
                                "Gains From Foreign Exchange",
                                "Subsidies",
                                "Other Non-Operating Income",
                                "Non-Operating Expenses",
                                "Interest Expenses PL True",
                                "Investment Loss on Equity Method",
                                "Loss From Foreign Exchange",
                                "Transaction Fees",
                                "Other Non-Operating Expenses",
                                "Ordinary Profit",
                                "Extraordinary Income",
                                "Extraordinary Loss",
                                "Net Income Before Taxes",
                                "Provision for Income Taxes",
                                "Net Income",
                                " ",
                                "Net Income/Starting Line",
                                "Depreciation/Depletion",
                                "Amortization of Goodwill",
                                "Depreciation and Amortization on Other",
                                "Increase (Decrease) in Provision",
                                "Interest and Dividend Income CF",
                                "Interest Expenses CF",
                                "Share of Loss (Profit) of Entities Accounted for Using Equity Method",
                                "Other Loss (Gain)",
                                "Decrease (Increase) in Trade Receivables",
                                "Decrease (Increase) in Inventories",
                                "Increase (Decrease) in Advances Received",
                                "Decrease (Increase) in Prepaid Expenses",
                                "Increase (Decrease) in Trade Payables",
                                "Increase (Decrease) in Retirement Benefit Liabilities",
                                "Increase (Decrease) in Accounts Payable - Other, and Accrued Expenses",
                                "Other, Net",
                                "Subtotal",
                                "Interest and Dividends Received",
                                "Interest Paid",
                                "Proceeds from Insurance Income",
                                "Compensation Paid for Damage",
                                "Cash Taxes Paid",
                                "Other Cash from Operating Activities",
                                "Cash from Operating Activities",
                                "Purchase of Securities",
                                "Proceeds from Sale of Securities",
                                "Purchase of Property, Plant and Equipment",
                                "Proceeds from Sale of Property, Plant and Equipment",
                                "Purchase of Intangible Assets",
                                "Proceeds from Sale of Intangible Assets",
                                "Purchase of Investment Securities",
                                "Proceeds from Sale of Investment Securities",
                                "Purchase of Shares of Subsidiaries Resulting in Change in Scope of Consolidation",
                                "Proceeds from Sale of Shares of Subsidiaries Resulting in Change in Scope of Consolidation",
                                "Loans Advances",
                                "Proceeds from Collection of Loans Receivable",
                                "Payments into Time Deposits",
                                "Proceeds from Withdrawal of Time Deposits",
                                "Payments of Leasehold and Guarantee Deposits",
                                "Proceeds from Refund of Leasehold and Guarantee Deposits",
                                "Payments for Acquisition of Businesses",
                                "Proceeds from Sale of Businesses",
                                "Other Cash from Investing Activities",
                                "Cash from Investing Activities",
                                "Proceeds from Short Term Borrowings",
                                "Repayments of Short Term Borrowings",
                                "Proceeds from Long Term Borrowings",
                                "Repayments of Long Term Borrowings",
                                "Proceeds from Issuance of Bonds",
                                "Redemption of Bonds",
                                "Proceeds from Issuance of Shares",
                                "Proceeds from Share Issuance to Non-Controlling Shareholders",
                                "Proceeds from Sale of Shares of Subsidiaries Not Resulting in Change in Scope of Consolidation",
                                "Purchase of Shares of Subsidiaries Not Resulting in Change in Scope of Consolidation",
                                "Purchase of Treasury Shares",
                                "Proceeds from Sale of Treasury Shares",
                                "Repayments of Lease Liabilities",
                                "Dividends Paid",
                                "Dividends Paid to Non-Controlling Interests",
                                "Other Cash from Financing Activities",
                                "Cash from Financing Activities",
                                " ",
                                "Foreign Exchange Effects",
                                "Net Change in Cash",
                                "Ending Cash Balance"])

    df_jasa = df_jasa.rename(columns={"Notes Receivable/Accounts Receivable True":"Notes Receivable/Accounts Receivable", 
                                      "Total Inventory True":"Total Inventory",
                                      "Leasehold and Guarantee Deposits True":"Leasehold and Guarantee Deposits",
                                      "Notes Payable/Accounts Payable True":"Notes Payable/Accounts Payable",
                                      "Advances Received True":"Advances Received",
                                      "Total Revenue True":"Total Revenue",
                                      "Cost of Revenue, Total True":"Cost of Revenue, Total",
                                      "Gross Profit True":"Gross Profit",
                                      "Logistics Costs True":"Logistics Costs",
                                      "Personnel Expenses True":"Personnel Expenses",
                                      "Depreciation / Amortization True":"Depreciation / Amortization",
                                      "Operating Income True":"Operating Income",
                                      "Interest Expenses PL True":"Interest Expenses PL",
                                      "Valuation/Exchange Differences, etc.":"Other Comprehensive Income, Total"})

    # 0より大きく10より小さい値を欠損値に置換
    df_jasa = df_jasa.apply(lambda col: col.map(lambda x: np.nan if (x > 0 and x < 10) else x))

    # 全てのセルの値を下3桁が000になるように変換
    df_jasa = df_jasa.apply(lambda col: col.map(lambda x: np.floor(x / 1000) * 1000 if isinstance(x, (int, float)) else x))

    #df_jasa = df_jasa.drop("Unnamed: 0", axis = 1)

    #欠損値以外をint型に変換、欠損値は"N/A"という文字に変換
    df_jasa = df_jasa.fillna(-1).round().astype(np.float64)
    df_jasa = df_jasa.replace(-1, "N/A")

    return df_jasa


# 全ての情報を一括取得して、JASAフォーマットに変換する(use)
def JASA_all_NonConsolidated(df_info, df_key_list, filtered_df):
    def JASA_ca(df_info, df_key_list, filtered_df):
        jasa_ca_code = ["Cash and Short Term Investments", 
                    "Notes Receivable/Accounts Receivable",
                "Notes Receivable/Accounts Receivable 1",
                "Notes Receivable/Accounts Receivable 2",
                "Notes Receivable/Accounts Receivable 3",
                "Notes Receivable/Accounts Receivable 4",
                "Notes Receivable/Accounts Receivable 5",
                "Notes Receivable/Accounts Receivable 6",
                "Notes Receivable/Accounts Receivable 7",
                "Electronically Recorded Monetary Claims",
                "Total Inventory",
                "Inventory1", 
                "Inventory2",
                "Inventory3",
                "Inventory4",
                "Inventory",
                "Prepaid Expenses",
                "Trading Securities",
                "Total Current Assets"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") 
        join_df = join_df.fillna(0.1)
        df_jasa_ca = join_df[join_df["JASA"].isin(jasa_ca_code)]
        df_jasa_ca = df_jasa_ca.drop("要素ID", axis = 1)
        df_jasa_ca.iloc[:, 1:] = df_jasa_ca.iloc[:, 1:].astype(np.float64)
        df_jasa_ca = df_jasa_ca.groupby(["JASA"]).sum()

        inventory_values = []
        for column_name, item in df_jasa_ca.items():
            if item["Inventory1"] > 1:
                if item["Inventory3"] > 1:
                    temp = item["Inventory1"] + item["Inventory3"] + item["Inventory"]
                else:
                    temp = item["Inventory1"] + item["Inventory4"] + item["Inventory"]
            else:
                if item["Inventory3"] > 1:
                    temp = item["Inventory2"] + item["Inventory3"] + item["Inventory"]
                else:
                    temp = item["Inventory2"] + item["Inventory4"] + item["Inventory"]
            inventory_values.append(temp)

        df_jasa_ca.loc["Inventory True"] = inventory_values 

        total_inventory_values = []
        for column_name, item in df_jasa_ca.items():
            if item["Total Inventory"] > item["Inventory True"]: 
                temp = item["Total Inventory"]
                if column_name == str(1802):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00055-000:InventoriesForPFIAndOtherProjectsCA", str(column_name)])
                elif column_name == str(1803):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:UncompletedRealEstateDevelopmentProjectsCA", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:PFIProjectsAndOtherInventoriesCA", str(column_name)])
                elif column_name == str(1812):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00058-000:CostsOnDevelopmentProjectsInProgressCA", str(column_name)])
            else:
                temp = item["Inventory True"]
                if column_name == str(1802):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00055-000:InventoriesForPFIAndOtherProjectsCA", str(column_name)])
                elif column_name == str(1803):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:UncompletedRealEstateDevelopmentProjectsCA", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:PFIProjectsAndOtherInventoriesCA", str(column_name)])
                elif column_name == str(1812):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00058-000:CostsOnDevelopmentProjectsInProgressCA", str(column_name)])
            total_inventory_values.append(temp)

        df_jasa_ca.loc["Total Inventory True"] = total_inventory_values

                
        receivable_values = []
        for column_name, item in df_jasa_ca.items():
            if item["Notes Receivable/Accounts Receivable 1"] > 1:
                temp = item["Notes Receivable/Accounts Receivable 1"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            elif item["Notes Receivable/Accounts Receivable 2"] > 1:
                temp = item["Notes Receivable/Accounts Receivable 2"] + item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            elif item["Notes Receivable/Accounts Receivable 3"] > 1:
                temp = item["Notes Receivable/Accounts Receivable 3"] + item["Notes Receivable/Accounts Receivable 4"] + item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            elif item["Notes Receivable/Accounts Receivable 6"] > 1:
                temp = item["Notes Receivable/Accounts Receivable 6"] + item["Notes Receivable/Accounts Receivable 4"] + item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            elif item["Notes Receivable/Accounts Receivable 4"] > 1:
                temp = item["Notes Receivable/Accounts Receivable 4"] + item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            elif item["Notes Receivable/Accounts Receivable 5"] > 1:
                temp = item["Notes Receivable/Accounts Receivable 5"] + item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            else:
                temp = item["Notes Receivable/Accounts Receivable 7"] + item["Notes Receivable/Accounts Receivable"]
                if column_name == str(1963):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01575-000:NotesReceivableTradeReceivablesContractAssetsAndOtherCA", str(column_name)])
                elif column_name == str(6366):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E01569-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndContractAssets", str(column_name)])
                elif column_name == str(9104):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04236-000:NotesAndAccountsReceivableTradeCA", str(column_name)])
                elif column_name == str(1967):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00138-000:NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherContractAssetsCA", str(column_name)])
            receivable_values.append(temp)


        df_jasa_ca.loc["Notes Receivable/Accounts Receivable True"] = receivable_values

        # receivable_ca = []
        # for column_name, item in df_jasa_ca.items():
        #     temp = item["Notes Receivable/Accounts Receivable True"]
        #     if column_name == str(6532):
        #         temp += float(filtered_df.loc["jpcrp030000-asr_E32549-000:AccountsReceivableTradeAndContractAssetsCA", str(column_name)])
        #     receivable_ca.append(temp)
        
        # df_jasa_ca.loc["Notes Receivable/Accounts Receivable True"] = receivable_ca



        other_ca = []
        for column_name, item in df_jasa_ca.items():
            temp = item["Total Current Assets"] - item["Cash and Short Term Investments"] - item["Notes Receivable/Accounts Receivable True"] - item["Total Inventory True"] - item["Prepaid Expenses"] - item["Trading Securities"]
            other_ca.append(temp)

        
        df_jasa_ca.loc["Other Current Assets"] = other_ca

        df_jasa_ca = df_jasa_ca.reindex(index=["Total Current Assets", 
                                            "Cash and Short Term Investments",
                                            "Notes Receivable/Accounts Receivable True",
                                            "Total Inventory True",
                                            "Prepaid Expenses",
                                            "Trading Securities",
                                            "Other Current Assets",
                                            "Electronically Recorded Monetary Claims",
                                            "Notes Receivable/Accounts Receivable 1",
                                            "Notes Receivable/Accounts Receivable 2",
                                            "Notes Receivable/Accounts Receivable 3",
                                            "Notes Receivable/Accounts Receivable 4",
                                            "Notes Receivable/Accounts Receivable 5",
                                            "Notes Receivable/Accounts Receivable 6",
                                            "Notes Receivable/Accounts Receivable 7",
                                            "Notes Receivable/Accounts Receivable"])

        return df_jasa_ca

    def JASA_nca(df_info, df_key_list, filtered_df):
        jasa_nca_code = ["Total Non-Current Assets",
                        "Property/Plant/Equipment, Total - Net",
                        "Intangibles, Net",
                        "Goodwill, Net",
                        "Investments and Other Assets",
                        "Long Term Investments",
                        "Leasehold and Guarantee Deposits 1",
                        "Leasehold and Guarantee Deposits 2",
                        "Long Term Deposits",
                        "Deferred Tax Assets",
                        "Total Assets"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") #ここを消した場合、別記事業codeが入って、error(float + str)が出るはず。でもここが問題か？
        join_df = join_df.fillna(0.1)
        df_jasa_nca = join_df[join_df["JASA"].isin(jasa_nca_code)]
        df_jasa_nca = df_jasa_nca.drop("要素ID", axis = 1)
        df_jasa_nca.iloc[:, 1:] = df_jasa_nca.iloc[:, 1:].astype(np.float64)
        df_jasa_nca = df_jasa_nca.groupby(["JASA"]).sum()

        lease_values = []
        for column_name, item in df_jasa_nca.items():
            if item["Leasehold and Guarantee Deposits 1"] > 1:
                temp = item["Leasehold and Guarantee Deposits 1"]
            else:
                temp = item["Leasehold and Guarantee Deposits 2"]
            lease_values.append(temp)

        df_jasa_nca.loc["Leasehold and Guarantee Deposits True"] = lease_values

        intangibles_other_values = []
        for column_name, item in df_jasa_nca.items():
            temp = item["Intangibles, Net"] - item["Goodwill, Net"]
            intangibles_other_values.append(temp)

        df_jasa_nca.loc["Other Intangibles, Net"] = intangibles_other_values

        other_ncl = []
        for column_name, item in df_jasa_nca.items():
            temp = item["Total Non-Current Assets"] - item["Property/Plant/Equipment, Total - Net"] - item["Goodwill, Net"] - item["Other Intangibles, Net"] - item["Long Term Investments"] - item["Leasehold and Guarantee Deposits True"] - item["Deferred Tax Assets"]
            other_ncl.append(temp)
        
        df_jasa_nca.loc["Other Non-Current Assets"] = other_ncl

        df_jasa_nca = df_jasa_nca.reindex(index=["Total Non-Current Assets",
                                                "Property/Plant/Equipment, Total - Net",
                                                "Goodwill, Net",
                                                "Other Intangibles, Net",
                                                "Long Term Investments",
                                                "Leasehold and Guarantee Deposits True",
                                                "Deferred Tax Assets",
                                                "Other Non-Current Assets",
                                                "Total Assets"])

        return df_jasa_nca

    def JASA_cl(df_info, df_key_list, filtered_df):
        jasa_cl_code = ["Notes Payable/Accounts Payable",
                        "Notes Payable/Accounts Payable 1",
                    "Notes Payable/Accounts Payable 2",
                    "Notes Payable/Accounts Payable 3",
                    "Notes Payable/Accounts Payable 4",
                    "Notes Payable/Accounts Payable 5",
                    "Notes Payable/Accounts Payable 6",
                    "Notes Payable/Accounts Payable 7",
                    "Electronically Recorded Monetary Debt",
                    "Capital Leases",
                    "Short Term Debt",
                    "Current Port. of LT Debt",
                    "Advances Received",
                    "Total Current Liabilities"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        join_df = join_df.fillna(0.1)
        df_jasa_cl = join_df[join_df["JASA"].isin(jasa_cl_code)]
        df_jasa_cl = df_jasa_cl.drop("要素ID", axis = 1)
        df_jasa_cl.iloc[:, 1:] = df_jasa_cl.iloc[:, 1:].astype(np.float64)
        df_jasa_cl = df_jasa_cl.groupby(["JASA"]).sum()

        notes_payable_values = []
        for column_name, item in df_jasa_cl.items():
            if item["Notes Payable/Accounts Payable 1"] > 1:
                if item["Notes Payable/Accounts Payable 6"] > 1:
                    temp = item["Notes Payable/Accounts Payable 1"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
                else:
                    temp = item["Notes Payable/Accounts Payable 1"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
            elif item["Notes Payable/Accounts Payable 2"]:
                if item["Notes Payable/Accounts Payable 6"] > 1:
                    temp = item["Notes Payable/Accounts Payable 2"] + item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
                else:
                    temp = item["Notes Payable/Accounts Payable 2"] + item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
            elif item["Notes Payable/Accounts Payable 5"] > 1:
                if item["Notes Payable/Accounts Payable 6"] > 1:
                    temp = item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 5"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
                else:
                    temp = item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 5"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
            elif item["Notes Payable/Accounts Payable 3"] > 1:
                if item["Notes Payable/Accounts Payable 6"] > 1:
                    temp = item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
                elif item["Notes Payable/Accounts Payable 7"] > 1:
                    temp = item["Notes Payable/Accounts Payable 3"] + item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
            else:
                if item["Notes Payable/Accounts Payable 6"] > 1:
                    temp = item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 6"] + item["Notes Payable/Accounts Payable"]
                else:
                    temp = item["Notes Payable/Accounts Payable 4"] + item["Notes Payable/Accounts Payable 7"] + item["Notes Payable/Accounts Payable"]
            notes_payable_values.append(temp)

        df_jasa_cl.loc["Notes Payable/Accounts Payable True"] = notes_payable_values  

        # advances_values = []
        # for column_name, item in df_jasa_cl.items():
        #     if item["Advances Received 2"] > 1:
        #         temp = item["Advances Received 2"]
        #     else:
        #         temp = item["Advances Received 1"]
        #     advances_values.append(temp)

        # df_jasa_cl.loc["Advances Received True"] = advances_values

        advances_values = []
        for column_name, item in df_jasa_cl.items():
            temp = item["Advances Received"]
            if column_name == str(1802):
                temp += float(filtered_df.loc["jpcrp030000-asr_E00055-000:AdvancesReceivedOnConstructionContractsInProgressContractLiabilitiesCLCNS", str(column_name)])
            elif column_name == str(1812):
                temp += float(filtered_df.loc["jpcrp030000-asr_E00058-000:AdvancesReceivedOnConstructionProjectsInProgress", str(column_name)])

            advances_values.append(temp)

        df_jasa_cl.loc["Advances Received True"] = advances_values

        short_values = []
        for column_name, item in df_jasa_cl.items():
            if item["Short Term Debt"] < item["Total Current Liabilities"] - (item["Notes Payable/Accounts Payable True"] + item["Advances Received True"] + item["Capital Leases"]):
                temp = item["Short Term Debt"]
            else:
                temp = 0
            short_values.append(temp)

        df_jasa_cl.loc["Short Term Debt"] = short_values

        long_values = []
        for column_name, item in df_jasa_cl.items():
            if item["Current Port. of LT Debt"] < item["Total Current Liabilities"] - (item["Notes Payable/Accounts Payable True"] + item["Advances Received True"] + item["Capital Leases"] + item["Short Term Debt"]):
                temp = item["Current Port. of LT Debt"]
                if column_name == str(1802):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00055-000:CurrentPortionOfNonrecourseLoansCL", str(column_name)])
                elif column_name == str(1803):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:CurrentPortionOfNonRecourseBorrowingsCL", str(column_name)])
            else:
                temp = 0
                if column_name == str(1802):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00055-000:CurrentPortionOfNonrecourseLoansCL", str(column_name)])
                elif column_name == str(1803):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:CurrentPortionOfNonRecourseBorrowingsCL", str(column_name)])
            long_values.append(temp)

        df_jasa_cl.loc["Current Port. of LT Debt"] = long_values

        other_cl = []
        for column_name, item in df_jasa_cl.items():
            temp = item["Total Current Liabilities"] - item["Notes Payable/Accounts Payable True"] - item["Advances Received True"] - item["Capital Leases"] - item["Short Term Debt"] - item["Current Port. of LT Debt"]
            other_cl.append(temp)
        
        df_jasa_cl.loc["Other Current Liabilities"] = other_cl

        df_jasa_cl = df_jasa_cl.reindex(index=["Total Current Liabilities", 
                                            "Notes Payable/Accounts Payable True",
                                            "Advances Received True",
                                            "Capital Leases",
                                            "Short Term Debt",
                                            "Current Port. of LT Debt",
                                            "Other Current Liabilities",
                                            "Electronically Recorded Monetary Debt",
                                            "Notes Payable/Accounts Payable 1",
                                            "Notes Payable/Accounts Payable 2",
                                            "Notes Payable/Accounts Payable 3",
                                            "Notes Payable/Accounts Payable 4",
                                            "Notes Payable/Accounts Payable 5",
                                            "Notes Payable/Accounts Payable 6",
                                            "Notes Payable/Accounts Payable 7",
                                            "Notes Payable/Accounts Payable"]) 
        
        return df_jasa_cl


    def JASA_ncl(df_info, df_key_list, filtered_df):
        jasa_ncl_code = ["Total Non-Current Liabilities",
                        "Long Term Debt",
                        "Lease Liabilities",
                        "Retirement Benefit",
                        "Deferred Tax Liabilities",
                        "Asset Retirement Obligation",
                        "Bonds",
                        "Provision NCL, Total"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        join_df = join_df.fillna(0.1)
        df_jasa_ncl = join_df[join_df["JASA"].isin(jasa_ncl_code)]
        df_jasa_ncl = df_jasa_ncl.drop("要素ID", axis = 1)
        df_jasa_ncl.iloc[:, 1:] = df_jasa_ncl.iloc[:, 1:].astype(np.float64)
        df_jasa_ncl = df_jasa_ncl.groupby(["JASA"]).sum()
                
        long_term_debt = []
        for column_name, item in df_jasa_ncl.items():
            temp = item["Long Term Debt"]
            if column_name == str(1802):
                temp += float(filtered_df.loc["jpcrp030000-asr_E00055-000:NonrecourseLoansNCL", str(column_name)])
            elif column_name == str(1803):
                temp += float(filtered_df.loc["jpcrp030000-asr_E00053-000:NonRecourseBorrowingsNCL", str(column_name)])
            long_term_debt.append(temp)

        df_jasa_ncl.loc["Long Term Debt"] = long_term_debt

        other_ncl = []
        for column_name, item in df_jasa_ncl.items():
            temp = item["Total Non-Current Liabilities"] - item["Long Term Debt"] - item["Bonds"] - item["Lease Liabilities"] - item["Retirement Benefit"] - item["Deferred Tax Liabilities"] - item["Asset Retirement Obligation"]
            other_ncl.append(temp)

        df_jasa_ncl.loc["Other Non-Current Liabilities"] = other_ncl

        df_jasa_ncl = df_jasa_ncl.reindex(index=["Total Non-Current Liabilities",
                                                "Long Term Debt",
                                                "Bonds",
                                                "Lease Liabilities",
                                                "Retirement Benefit",
                                                "Deferred Tax Liabilities",
                                                "Asset Retirement Obligation",
                                                "Other Non-Current Liabilities",
                                                "Provision NCL, Total"])

        return df_jasa_ncl


    def JASA_equity_NonConsolidated(df_info, df_key_list, filtered_df):
        jasa_equity_code = ["Total Liabilities",
                            "Common Stock",
                            "Total, Additional Paid-In",
                            "Retained Earnings",
                            "Treasury Stock-Common",
                            "Total Shareholder's Equity",
                            "Valuation/Exchange Differences, etc.",
                            "Other Equity, Total",
                            "Net Assets",
                            "Total Liabilities & Net Assets"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        join_df = join_df.fillna(0.1)
        df_jasa_equity = join_df[join_df["JASA"].isin(jasa_equity_code)]
        df_jasa_equity = df_jasa_equity.drop_duplicates(subset="要素ID")
        df_jasa_equity = df_jasa_equity.drop("要素ID", axis = 1)
        df_jasa_equity.iloc[:, 1:] = df_jasa_equity.iloc[:, 1:].astype(np.float64)
        df_jasa_equity = df_jasa_equity.groupby(["JASA"]).sum()

        other_equity = []
        for column_name, item in df_jasa_equity.items():
            temp = item["Net Assets"] - item["Common Stock"] - item["Total, Additional Paid-In"] - item["Retained Earnings"] - item["Treasury Stock-Common"] - item["Valuation/Exchange Differences, etc."]
            other_equity.append(temp)

        df_jasa_equity.loc["Other Equity, Total"] = other_equity


        df_jasa_equity = df_jasa_equity.reindex(index=["Total Liabilities",
                                                        "Common Stock",
                                                        "Total, Additional Paid-In",
                                                        "Retained Earnings",
                                                        "Treasury Stock-Common",
                                                        "Valuation/Exchange Differences, etc.",
                                                        "Other Equity, Total",
                                                        "Net Assets",
                                                        "Total Liabilities & Net Assets",
                                                        "Total Shareholder's Equity"])

        return df_jasa_equity

    def JASA_revenue(df_info, df_key_list, filtered_df):
        jasa_revenue_code = ["Total Revenue 1",
                            "Total Revenue 2",
                            "Total Revenue 3",
                            "Total Revenue 4",
                            "Total Revenue 5",
                            "Cost of Revenue, Total 1",
                            "Cost of Revenue, Total 2",
                            "Gross Profit",
                            "Gross Profit 3"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        join_df = join_df.fillna(0.1)
        df_jasa_revenue = join_df[join_df["JASA"].isin(jasa_revenue_code)]
        df_jasa_revenue = df_jasa_revenue.drop("要素ID", axis = 1)
        df_jasa_revenue.iloc[:, 1:] = df_jasa_revenue.iloc[:, 1:].astype(np.float64)
        df_jasa_revenue = df_jasa_revenue.groupby(["JASA"]).sum()

        revenue_values = []
        for column_name, item in df_jasa_revenue.items():
            if (column_name == "9706" or column_name == "6183" or column_name == "6028" or 
        column_name == "8572" or column_name == "8515" or column_name == "3563" or 
        column_name == "9069" or column_name == "9301" or column_name == "9324" or 
        column_name == "9143" or column_name == "9733" or column_name == "8905" or 
        column_name == "8801" or column_name == "3289" or column_name == "3231" or 
        column_name == "8830" or column_name == "8802" or column_name == "8804" or 
        column_name == "8609" or column_name == "8616" or column_name == "7453" or 
        column_name == "9304" or column_name == "9302" or column_name == "9502" or 
        column_name == "9506" or column_name == "9513" or column_name == "9508" or 
        column_name == "9504" or column_name == "3003"):
                temp = item["Total Revenue 3"]
            elif (column_name == "9602" or column_name == "9206"):
                temp = item["Total Revenue 4"]
            elif (column_name == "6532" or column_name == "8252"):
                temp = item["Total Revenue 2"]
            else:
                temp = item["Total Revenue 1"]
            revenue_values.append(temp)

        df_jasa_revenue.loc["Total Revenue True"] = revenue_values
            
        cost_of_revenue_values = []
        for column_name, item in df_jasa_revenue.items():
            if (column_name == "9602" or column_name == "9069" or column_name == "9301" or 
        column_name == "9324" or column_name == "9143" or column_name == "9733" or 
        column_name == "8905" or column_name == "8801" or column_name == "3289" or 
        column_name == "3231" or column_name == "8830" or column_name == "8802" or 
        column_name == "8804" or column_name == "7453" or column_name == "9304" or 
        column_name == "3003"):
                temp = item["Cost of Revenue, Total 2"]
            else:
                temp = item["Cost of Revenue, Total 1"]
            cost_of_revenue_values.append(temp)

        df_jasa_revenue.loc["Cost of Revenue, Total True"] = cost_of_revenue_values 


        gross_values = []
        for column_name, item in df_jasa_revenue.items():
            if (column_name == "9706" or column_name == "9069" or column_name == "9301" or 
        column_name == "9324" or column_name == "9143" or column_name == "9733" or 
        column_name == "9206" or column_name == "8905" or column_name == "3289" or 
        column_name == "3231" or column_name == "8802" or column_name == "8804" or 
        column_name == "7453" or column_name == "9304" or column_name == "9302" or 
        column_name == "3003"):
                temp = item["Gross Profit 3"]
            else:
                temp = item["Gross Profit"]
            gross_values.append(temp)

        df_jasa_revenue.loc["Gross Profit True"] = gross_values 


        df_jasa_revenue = df_jasa_revenue.reindex(index = ["Total Revenue True",
                                                        "Total Revenue 1",
                                                        "Total Revenue 2",
                                                        "Total Revenue 3",
                                                        "Total Revenue 4",
                                                        "Total Revenue 5",
                                                        "Cost of Revenue, Total True",
                                                    "Cost of Revenue, Total 1",
                                                    "Cost of Revenue, Total 2",
                                                    "Gross Profit True",
                                                    "Gross Profit",
                                                    "Gross Profit 3"])

        return df_jasa_revenue

    def JASA_pl_NonConsolidated(df_info, df_key_list, filtered_df):
        jasa_pl_code = ["Total Operating Expenses",
                        "Selling/General/Admin. Expenses, Total",
                        "Personnel Expenses",
                        "Personnel Expenses a",
                        "Research & Development",
                        "Advertising Expenses",
                        "Outsourcing Cost",
                        "Rents",
                        "Depreciation / Amortization",
                        "Depreciation / Amortization a",
                        "Logistics Costs 1",
                        "Logistics Costs 2",
                        "Operating Income 1",
                        "Operating Income 2",
                        "Non-Operating Income",
                        "Interest and Dividend Income PL",
                        "Interest and Dividend Income PL 2",
                        "Interest and Dividend Income PL 3",
                        "Investment Gains on Equity Method",
                        "Gains From Foreign Exchange",
                        "Rental Income",
                        "Gains From Sale of Assets",
                        "Subsidies",
                        "Non-Operating Expenses",
                        "Interest Expenses PL",
                        "Interest Expenses PL 2",
                        "Loss From Sale of Assets",
                        "Investment Loss on Equity Method",
                        "Rental Expenses",
                        "Loss From Foreign Exchange",
                        "Transaction Fees",
                        "Ordinary Profit",
                        "Extraordinary Income",
                        "Extraordinary Loss",
                        "Net Income Before Taxes",
                        "Provision for Income Taxes",
                        "Net Income"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") #ここを消した場合、別記事業codeが入って、error(float + str)が出るはず。でもここが問題か？
        join_df = join_df.fillna(0.1)
        df_jasa_pl = join_df[join_df["JASA"].isin(jasa_pl_code)]
        df_jasa_pl = df_jasa_pl.drop_duplicates(subset="要素ID")
        df_jasa_pl = df_jasa_pl.drop("要素ID", axis = 1)
        df_jasa_pl.iloc[:, 1:] = df_jasa_pl.iloc[:, 1:].astype(np.float64)
        df_jasa_pl = df_jasa_pl.groupby(["JASA"]).sum()
        filtered_df.iloc[:, 1:] = filtered_df.iloc[:, 1:].astype(np.float64, errors="ignore")

        interest_pl_values = []
        for column_name, item in df_jasa_pl.items():
            if (item["Interest and Dividend Income PL"] <= -50 or item["Interest and Dividend Income PL"] >= 50) and item["Interest and Dividend Income PL"] < item["Non-Operating Income"]:
                temp = item["Interest and Dividend Income PL"]
            elif (item["Interest and Dividend Income PL 2"] <= -50 or item["Interest and Dividend Income PL 2"] >= 50) and item["Interest and Dividend Income PL 2"] < item["Non-Operating Income"]:
                temp = item["Interest and Dividend Income PL 2"]
            else:
                temp = item["Interest and Dividend Income PL 3"]
            interest_pl_values.append(temp)

        df_jasa_pl.loc["Interest and Dividend Income PL True"] = interest_pl_values

        interest_pl_losses = []
        for column_name, item in df_jasa_pl.items():
            if item["Interest Expenses PL"] <= -50 or item["Interest Expenses PL"] >= 50:
                temp = item["Interest Expenses PL"]
            else:
                temp = item["Interest Expenses PL 2"]
            interest_pl_losses.append(temp)

        df_jasa_pl.loc["Interest Expenses PL True"] = interest_pl_losses

        research_values = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Research & Development"]
            try:
                # ここで何らかの処理を行う
                if column_name == str(2229):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E25303-000:ResearchAndDevelopmentExpensesGeneralAndAdministrativeExpenses", str(column_name)])
                elif column_name == str(9783):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04939-000:ResearchAndDevelopmentCostsGeneralAndAdministrativeExpenses", str(column_name)])
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04939-000:ResearchAndDevelopmentCostsManufacturingCostForCurrentPeriod", str(column_name)])
            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            research_values.append(temp)

        df_jasa_pl.loc["Research & Development"] = research_values

        outsourcing_cost = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Outsourcing Cost"]
            try:
                # ここで何らかの処理を行う
                if column_name == str(9470):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00707-000:WorkConsignmentExpenses", str(column_name)])
                elif column_name == str(9202):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04273-000:OutsourcingFeeSGA", str(column_name)])
                elif column_name == str(9532):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04520-000:ConsignmentWorkExpensesSGA", str(column_name)])
                elif column_name == str(9513):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04510-000:ConsignmentCostSGA", str(column_name)])
            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            outsourcing_cost.append(temp)

        df_jasa_pl.loc["Outsourcing Cost"] = outsourcing_cost

        advertising_cost = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Advertising Expenses"]
            try:
                # ここで何らかの処理を行う
                if column_name == str(2613):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00434-000:AdvertisementSGA", str(column_name)])
                elif column_name == str(2267):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E00406-000:SalesPromotionExpensesSGA", str(column_name)])
                elif column_name == str(3382):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E03462-000:AdvertisingAndDecorationExpenses", str(column_name)])
                elif column_name == str(4676):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E04462-000:AdvertisementSellingExpenses", str(column_name)])
            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            advertising_cost.append(temp)

        df_jasa_pl.loc["Advertising Expenses"] = advertising_cost

        rent = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Rents"]
            try:
                if column_name == str(9302):
                    temp -= float(filtered_df.loc["jppfs_cor:RentExpensesSGA", str(column_name)])
                elif column_name == str(7816):
                    temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:Rents", str(column_name)])
            except KeyError:
                print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            rent.append(temp)

        df_jasa_pl.loc["Rents"] = rent

        depreciation_values = []
        for column_name, item in df_jasa_pl.items():
            if item["Depreciation / Amortization"] <= -50 or item["Depreciation / Amortization"] >= 50:
                temp = item["Depreciation / Amortization"] + item["Depreciation / Amortization a"]
                try:
                    if column_name == str(5902):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:AmortizationOfGoodwill", str(column_name)])
                    elif column_name == str(6532):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E32549-000:DepreciationAndAmortizationSGA", str(column_name)])
                except KeyError:
                    print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            else:
                temp = item["Depreciation / Amortization a"]
                try:
                    if column_name == str(5902):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:AmortizationOfGoodwill", str(column_name)])
                    elif column_name == str(6532):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E32549-000:DepreciationAndAmortizationSGA", str(column_name)])

                except KeyError:
                    print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            depreciation_values.append(temp)

        df_jasa_pl.loc["Depreciation / Amortization True"] = depreciation_values 
        
        personnel_values = []
        for column_name, item in df_jasa_pl.items():
            if item["Personnel Expenses"] <= -50 or item["Personnel Expenses"] >= 50:
                temp = item["Personnel Expenses"] + item["Personnel Expenses a"]
                try:
                    if column_name == str(7972):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02371-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E02371-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                    elif column_name == str(2168):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E05729-000:EmployeePayrollsAndCompensationAndOtherSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E05729-000:EmployeePayrollsAndCompensationAndOtherSGA", str(column_name)])
                    elif column_name == str(9278):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E34102-000:SalariesOfPartTimeEmployeesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E34102-000:SalariesOfPartTimeEmployeesSGA", str(column_name)])
                    elif column_name == str(2229):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E25303-000:PayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E25303-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(9861):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03153-000:CostForPartTimersSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E03153-000:CostForPartTimersSGA", str(column_name)])
                    elif column_name == str(7581):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03305-000:EmployeesPayrollsAndBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E03305-000:EmployeesPayrollsAndBonusesSGA", str(column_name)])
                    elif column_name == str(4934):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E36046-000:RetirementBenefitExpense", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E36046-000:RetirementBenefitExpense", str(column_name)])
                    elif column_name == str(4031):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00789-000:PayrollsAllowancesAndBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00789-000:PayrollsAllowancesAndBonusesSGA", str(column_name)])
                    elif column_name == str(6753):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01773-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01773-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                    elif column_name == str(6367):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01570-000:DirectorsAndEmployeePayrollsAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01570-000:DirectorsAndEmployeePayrollsAndAllowancesSGA", str(column_name)])
                    elif column_name == str(9301):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                    elif column_name == str(5929):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01385-000:ProvisionForEmployeeCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01385-000:ProvisionForEmployeeCompensationSGA", str(column_name)])
                    elif column_name == str(3407):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00877-000:SalariesAndBenefitsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                    elif column_name == str(3405):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00876-000:PayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00876-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(7269):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02167-000:PayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E02167-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(8905):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E04002-000:ProvisionForDirectorsRemunerationBasedOnPerformanceSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E04002-000:ProvisionForDirectorsRemunerationBasedOnPerformanceSGA", str(column_name)])
                    elif column_name == str(2871):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00446-000:DirectorsCompensationEmployeesSalariesBonusesAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00446-000:DirectorsCompensationEmployeesSalariesBonusesAndAllowancesSGA", str(column_name)])
                    elif column_name == str(2607):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00431-000:EmployeePayrollsAndAllowances", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00431-000:EmployeePayrollsAndAllowances", str(column_name)])
                    elif column_name == str(2004):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00348-000:EmployeeSalariesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00348-000:EmployeeSalariesSGA", str(column_name)])
                    elif column_name == str(6332):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(6508):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01744-000:BonusesAndProvisionForBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01744-000:BonusesAndProvisionForBonusesSGA", str(column_name)])
                    elif column_name == str(6368):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01571-000:EmployeePayrollsAllowancesAndCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01571-000:EmployeePayrollsAllowancesAndCompensationSGA", str(column_name)])
                    elif column_name == str(1301):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:SalesStaffPayrollsAndAllowancesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:OfficeStaffPayrollsAndAllowancesGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(5706):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00024-000:BonusAndRetirementPaymentsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00024-000:BonusAndRetirementPaymentsSGA", str(column_name)])
                    elif column_name == str(9304):
                        temp -= float(filtered_df.loc["jppfs_cor:PersonalExpensesSGA", str(column_name)])
                        #print(filtered_df.loc["jppfs_cor:PersonalExpensesSGA", str(column_name)])
                    elif column_name == str(9302):
                        temp -= float(filtered_df.loc["jppfs_cor:SalariesAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jppfs_cor:SalariesAndAllowancesSGA", str(column_name)])
                    elif column_name == str(2281):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00335-000:PayrollsAndOtherAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00335-000:PayrollsAndOtherAllowancesSGA", str(column_name)])
                    elif column_name == str(2264):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00331-000:EmployeesSalariesAndBonusesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00331-000:EmployeesSalariesAndBonusesGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(2270):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E23202-000:SalariesGeneralAndAdministrativeExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E23202-000:OtherGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E23202-000:SalariesGeneralAndAdministrativeExpenses", str(column_name)])
                    elif column_name == str(5333):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01137-000:PayrollsAndCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01137-000:PayrollsAndCompensationSGA", str(column_name)])
                    elif column_name == str(3382):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03462-000:SalariesAndWages", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E03462-000:SalariesAndWages", str(column_name)])
                    elif column_name == str(5902):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:ProvisionForManagementBoardIncentivePlanTrust", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:ProvisionForEmployeeStockOwnershipPlanTrust", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(6861):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01967-000:DirectorsCompensationAndEmployeeSalariesAllowancesAndCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01967-000:DirectorsCompensationAndEmployeeSalariesAllowancesAndCompensationSGA", str(column_name)])
                    elif column_name == str(7816):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:ProvisionForDirectorSPerformanceLinkedIncentiveCompensationSGA", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:ProvisionForEmployeeSPerformanceLinkedIncentiveCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(5233):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01130-000:PersonnelExpenditureSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01130-000:PersonnelExpenditureSGA", str(column_name)])
                    elif column_name == str(5232):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01127-000:PayrollsAndBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01127-000:PayrollsAndBonusesSGA", str(column_name)])
                    elif column_name == str(4043):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00768-000:SalariesBonusesAndAllowancesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00768-000:SalariesBonusesAndAllowancesGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(3191):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E30501-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(9206):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E26084-000:PayrollsAndAllowancesSGA", str(column_name)])

                    
                except KeyError:
                    print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")


            else:
                temp = item["Personnel Expenses a"]
                try:
                    if column_name == str(7972):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02371-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E02371-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                    elif column_name == str(2168):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E05729-000:EmployeePayrollsAndCompensationAndOtherSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E05729-000:EmployeePayrollsAndCompensationAndOtherSGA", str(column_name)])
                    elif column_name == str(9278):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E34102-000:SalariesOfPartTimeEmployeesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E34102-000:SalariesOfPartTimeEmployeesSGA", str(column_name)])
                    elif column_name == str(2229):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E25303-000:PayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E25303-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(9861):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03153-000:CostForPartTimersSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E03153-000:CostForPartTimersSGA", str(column_name)])
                    elif column_name == str(7581):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03305-000:EmployeesPayrollsAndBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E03305-000:EmployeesPayrollsAndBonusesSGA", str(column_name)])
                    elif column_name == str(4934):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E36046-000:RetirementBenefitExpense", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E36046-000:RetirementBenefitExpense", str(column_name)])
                    elif column_name == str(4031):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00789-000:PayrollsAllowancesAndBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00789-000:PayrollsAllowancesAndBonusesSGA", str(column_name)])
                    elif column_name == str(6753):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01773-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01773-000:EmployeePayrollsAndAllowancesSGA", str(column_name)])
                    elif column_name == str(6367):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01570-000:DirectorsAndEmployeePayrollsAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01570-000:DirectorsAndEmployeePayrollsAndAllowancesSGA", str(column_name)])
                    elif column_name == str(9301):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                    elif column_name == str(5929):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01385-000:ProvisionForEmployeeCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01385-000:ProvisionForEmployeeCompensationSGA", str(column_name)])
                    elif column_name == str(3407):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00877-000:SalariesAndBenefitsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E04283-000:CompensationAndPayrollsSGA", str(column_name)])
                    elif column_name == str(3405):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00876-000:PayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00876-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(7269):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02167-000:PayrollsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E02167-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(8905):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E04002-000:ProvisionForDirectorsRemunerationBasedOnPerformanceSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E04002-000:ProvisionForDirectorsRemunerationBasedOnPerformanceSGA", str(column_name)])
                    elif column_name == str(2871):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00446-000:DirectorsCompensationEmployeesSalariesBonusesAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00446-000:DirectorsCompensationEmployeesSalariesBonusesAndAllowancesSGA", str(column_name)])
                    elif column_name == str(2607):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00431-000:EmployeePayrollsAndAllowances", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00431-000:EmployeePayrollsAndAllowances", str(column_name)])
                    elif column_name == str(2004):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00348-000:EmployeeSalariesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00348-000:EmployeeSalariesSGA", str(column_name)])
                    elif column_name == str(6332):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(6508):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01744-000:BonusesAndProvisionForBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01744-000:BonusesAndProvisionForBonusesSGA", str(column_name)])
                    elif column_name == str(6368):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01571-000:EmployeePayrollsAllowancesAndCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01571-000:EmployeePayrollsAllowancesAndCompensationSGA", str(column_name)])
                    elif column_name == str(1301):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:SalesStaffPayrollsAndAllowancesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:OfficeStaffPayrollsAndAllowancesGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(5706):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00024-000:BonusAndRetirementPaymentsSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00024-000:BonusAndRetirementPaymentsSGA", str(column_name)])
                    elif column_name == str(9304):
                        temp -= float(filtered_df.loc["jppfs_cor:PersonalExpensesSGA", str(column_name)])
                        #print(filtered_df.loc["jppfs_cor:PersonalExpensesSGA", str(column_name)])
                    elif column_name == str(9302):
                        temp -= float(filtered_df.loc["jppfs_cor:SalariesAndAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jppfs_cor:SalariesAndAllowancesSGA", str(column_name)])
                    elif column_name == str(2281):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00335-000:PayrollsAndOtherAllowancesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E00335-000:PayrollsAndOtherAllowancesSGA", str(column_name)])
                    elif column_name == str(2264):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00331-000:EmployeesSalariesAndBonusesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00331-000:EmployeesSalariesAndBonusesGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(2270):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E23202-000:SalariesGeneralAndAdministrativeExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E23202-000:OtherGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E23202-000:SalariesGeneralAndAdministrativeExpenses", str(column_name)])
                    elif column_name == str(5333):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01137-000:PayrollsAndCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01137-000:PayrollsAndCompensationSGA", str(column_name)])
                    elif column_name == str(3382):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03462-000:SalariesAndWages", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E03462-000:SalariesAndWages", str(column_name)])
                    elif column_name == str(5902):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:ProvisionForManagementBoardIncentivePlanTrust", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01354-000:ProvisionForEmployeeStockOwnershipPlanTrust", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(6861):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01967-000:DirectorsCompensationAndEmployeeSalariesAllowancesAndCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01967-000:DirectorsCompensationAndEmployeeSalariesAllowancesAndCompensationSGA", str(column_name)])
                    elif column_name == str(7816):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:ProvisionForDirectorSPerformanceLinkedIncentiveCompensationSGA", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E31070-000:ProvisionForEmployeeSPerformanceLinkedIncentiveCompensationSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(5233):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01130-000:PersonnelExpenditureSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01130-000:PersonnelExpenditureSGA", str(column_name)])
                    elif column_name == str(5232):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01127-000:PayrollsAndBonusesSGA", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01127-000:PayrollsAndBonusesSGA", str(column_name)])
                    elif column_name == str(4043):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00768-000:SalariesBonusesAndAllowancesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00768-000:SalariesBonusesAndAllowancesGeneralAndAdministrativeExpenses", str(column_name)])
                        #print(filtered_df.loc["jpcrp030000-asr_E01537-000:DirectorsCompensationsSalariesAllowancesBonusesAndWelfareExpensesSGA", str(column_name)])
                    elif column_name == str(3191):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E30501-000:PayrollsSGA", str(column_name)])
                    elif column_name == str(9206):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E26084-000:PayrollsAndAllowancesSGA", str(column_name)])

                except KeyError:
                    print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            personnel_values.append(temp)

        df_jasa_pl.loc["Personnel Expenses True"] = personnel_values


        logistics_costs = []
        for column_name, item in df_jasa_pl.items():
            if item["Logistics Costs 1"] >= 50:
                temp = item["Logistics Costs 1"]
                try:
                    if column_name == str(8086):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02688-000:TransportationExpenses2SGA", str(column_name)])
                    elif column_name == str(6367):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01570-000:ProductFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(8173):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03052-000:DistributionExpenditureSGA", str(column_name)])
                    elif column_name == str(2211):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00374-000:FreightageAndStorageFeeSGA", str(column_name)])
                    elif column_name == str(9069):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E04179-000:UnderPaymentFareSGA", str(column_name)])
                    elif column_name == str(5332):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01138-000:DistributionExpenses", str(column_name)])
                    elif column_name == str(5932):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E26831-000:PackingMaterialsAndFreightageExpenditureSGA", str(column_name)])
                    elif column_name == str(3405):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00876-000:FreightageAndWarehouseExpensesSGA", str(column_name)])
                    elif column_name == str(7269):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02167-000:ShippingFee", str(column_name)])
                    elif column_name == str(2602):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00428-000:ProductsFreightageAndStorageFeeSGA", str(column_name)])
                    elif column_name == str(2613):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00434-000:ProductsFreightageFeeSGA", str(column_name)])
                    elif column_name == str(2004):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00348-000:ShippingAndDeliveryExpensesSGA", str(column_name)])
                    elif column_name == str(1301):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:ShippingAndDeliveryExpensesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:StorageExpensesSGA", str(column_name)])
                    elif column_name == str(7538):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02818-000:ShippingBountyAndCompletedDeliveryBountySGA", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02818-000:FreightExpensesSGA", str(column_name)])
                    elif column_name == str(8030):
                        temp += float(filtered_df.loc["jppfs_cor:SalesCommissionSGA", str(column_name)])
                    elif column_name == str(7453):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03248-000:DistributionExpensesAndTransportationCostsSGA", str(column_name)])
                    elif column_name == str(3315):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00030-000:OceanFreightSGA", str(column_name)])
                    elif column_name == str(3106):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00528-000:FreightOutExpenseWarehousingCostAndPackingExpenseSGA", str(column_name)])
                    elif column_name == str(2296):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E32069-000:ShippingAndDeliveryExpensesSGA", str(column_name)])
                    elif column_name == str(2281):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00335-000:PackingAndFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(1383):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E25969-000:PackingAndFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(5698):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E27868-000:TransportationSundryExpensesSGA", str(column_name)])
                    elif column_name == str(3864):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00644-000:PackingAndFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(3865):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00645-000:TransportationExpenditureSGA", str(column_name)])
                    elif column_name == str(5233):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01130-000:SalesFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(7513):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03270-000:TransportationCostsSGA", str(column_name)])
                    elif column_name == str(1925):
                        temp += float(filtered_df.loc["jppfs_cor:SalesCommissionSGA", str(column_name)])
                    elif column_name == str(3864):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00644-000:PackingAndFreightageExpensesSGA", str(column_name)])
 


                except KeyError:
                    print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")


            else:
                temp = item["Logistics Costs 2"]
                try:
                    if column_name == str(8086):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02688-000:TransportationExpenses2SGA", str(column_name)])
                    elif column_name == str(6367):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01570-000:ProductFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(8173):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03052-000:DistributionExpenditureSGA", str(column_name)])
                    elif column_name == str(2211):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00374-000:FreightageAndStorageFeeSGA", str(column_name)])
                    elif column_name == str(9069):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E04179-000:UnderPaymentFareSGA", str(column_name)])
                    elif column_name == str(5332):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01138-000:DistributionExpenses", str(column_name)])
                    elif column_name == str(5932):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E26831-000:PackingMaterialsAndFreightageExpenditureSGA", str(column_name)])
                    elif column_name == str(3405):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00876-000:FreightageAndWarehouseExpensesSGA", str(column_name)])
                    elif column_name == str(7269):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02167-000:ShippingFee", str(column_name)])
                    elif column_name == str(2602):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00428-000:ProductsFreightageAndStorageFeeSGA", str(column_name)])
                    elif column_name == str(2613):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00434-000:ProductsFreightageFeeSGA", str(column_name)])
                    elif column_name == str(2004):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00348-000:ShippingAndDeliveryExpensesSGA", str(column_name)])
                    elif column_name == str(1301):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:ShippingAndDeliveryExpensesSellingExpenses", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00012-000:StorageExpensesSGA", str(column_name)])
                    elif column_name == str(7538):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02818-000:ShippingBountyAndCompletedDeliveryBountySGA", str(column_name)])
                        temp += float(filtered_df.loc["jpcrp030000-asr_E02818-000:FreightExpensesSGA", str(column_name)])
                    elif column_name == str(8030):
                        temp += float(filtered_df.loc["jppfs_cor:SalesCommissionSGA", str(column_name)])
                    elif column_name == str(7453):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03248-000:DistributionExpensesAndTransportationCostsSGA", str(column_name)])
                    elif column_name == str(3315):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00030-000:OceanFreightSGA", str(column_name)])
                    elif column_name == str(3106):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00528-000:FreightOutExpenseWarehousingCostAndPackingExpenseSGA", str(column_name)])
                    elif column_name == str(2296):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E32069-000:ShippingAndDeliveryExpensesSGA", str(column_name)])
                    elif column_name == str(2281):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00335-000:PackingAndFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(1383):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E25969-000:PackingAndFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(5698):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E27868-000:TransportationSundryExpensesSGA", str(column_name)])
                    elif column_name == str(3864):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00644-000:PackingAndFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(3865):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00645-000:TransportationExpenditureSGA", str(column_name)])
                    elif column_name == str(5233):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E01130-000:SalesFreightageExpensesSGA", str(column_name)])
                    elif column_name == str(7513):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E03270-000:TransportationCostsSGA", str(column_name)])
                    elif column_name == str(1925):
                        temp += float(filtered_df.loc["jppfs_cor:SalesCommissionSGA", str(column_name)])
                    elif column_name == str(3864):
                        temp += float(filtered_df.loc["jpcrp030000-asr_E00644-000:PackingAndFreightageExpensesSGA", str(column_name)])
 
                except KeyError:
                    print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")

            logistics_costs.append(temp)

        df_jasa_pl.loc["Logistics Costs True"] = logistics_costs

        operating_values = []
        for column_name, item in df_jasa_pl.items():
            if item["Operating Income 1"] <= -50 or item["Operating Income 1"] >= 50:
                temp = item["Operating Income 1"]
            else :
                temp = item["Operating Income 2"]
            operating_values.append(temp)  

        df_jasa_pl.loc["Operating Income True"] = operating_values  

        other_sga = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Selling/General/Admin. Expenses, Total"] - item["Logistics Costs True"] - item["Personnel Expenses True"] - item["Advertising Expenses"] - item["Outsourcing Cost"] - item["Rents"] - item["Research & Development"] - item["Depreciation / Amortization True"]
            other_sga.append(temp)
        df_jasa_pl.loc["Other Selling/General/Admin. Expenses"] = other_sga

        other_noi = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Non-Operating Income"] - item["Interest and Dividend Income PL True"] - item["Investment Gains on Equity Method"] - item["Gains From Foreign Exchange"] - item["Rental Income"] - item["Subsidies"]
            other_noi.append(temp)
        df_jasa_pl.loc["Other Non-Operating Income"] = other_noi

        other_noe = []
        for column_name, item in df_jasa_pl.items():
            temp = item["Non-Operating Expenses"] - item["Interest Expenses PL True"] - item["Investment Loss on Equity Method"] - item["Loss From Foreign Exchange"] - item["Rental Expenses"] - item["Transaction Fees"]
            other_noe.append(temp)
        df_jasa_pl.loc["Other Non-Operating Expenses"] = other_noe

        df_jasa_pl = df_jasa_pl.reindex(index=["Total Operating Expenses",
                                                    "Selling/General/Admin. Expenses, Total",
                                                    "Logistics Costs True",
                                                    "Personnel Expenses True",
                                                    "Advertising Expenses",
                                                    "Outsourcing Cost",
                                                    "Rents",
                                                    "Research & Development",
                                                    "Depreciation / Amortization True",
                                                    "Other Selling/General/Admin. Expenses",
                                                    "Operating Income True",
                                                    "Non-Operating Income",
                                                    "Interest and Dividend Income PL True",
                                                    "Gains From Foreign Exchange",
                                                    "Rental Income",
                                                    "Subsidies",
                                                    "Other Non-Operating Income",
                                                    "Non-Operating Expenses",
                                                    "Interest Expenses PL True",
                                                    "Loss From Foreign Exchange",
                                                    "Rental Expenses",
                                                    "Transaction Fees",
                                                    "Other Non-Operating Expenses",
                                                    "Ordinary Profit",
                                                    "Extraordinary Income",
                                                    "Extraordinary Loss",
                                                    "Net Income Before Taxes",
                                                    "Provision for Income Taxes",
                                                    "Net Income"])


        return df_jasa_pl

    # 営業CF
    def JASA_opcf_NonConsolidated(df_info, df_key_list, filtered_df):
        jasa_opcf_code = ["Net Income Before Taxes",
                        "Depreciation/Depletion",
                        "Amortization of Goodwill",
                        "Depreciation and Amortization on Other",
                        "Increase (Decrease) in Provision",
                        "Interest and Dividend Income CF",
                        "Interest Expenses CF",
                        "Other Loss (Gain)",
                        "Decrease (Increase) in Trade Receivables",
                        "Decrease (Increase) in Inventories",
                        "Increase (Decrease) in Advances Received",
                        "Decrease (Increase) in Inventories",
                        "Increase (Decrease) in Advances Received",
                        "Decrease (Increase) in Prepaid Expenses",
                        "Increase (Decrease) in Trade Payables",
                        "Increase (Decrease) in Accounts Payable - Other, and Accrued Expenses",
                        "Subtotal",
                        "Interest and Dividends Received",
                        "Interest Paid",
                        "Proceeds from Insurance Income",
                        "Compensation Paid for Damage",
                        "Cash Taxes Paid",
                        "Cash from Operating Activities"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") 
        join_df = join_df.fillna(0.1)
        df_jasa_opcf = join_df[join_df["JASA"].isin(jasa_opcf_code)]
        df_jasa_opcf = df_jasa_opcf.drop("要素ID", axis = 1)
        df_jasa_opcf.iloc[:, 1:] = df_jasa_opcf.iloc[:, 1:].astype(np.float64)
        df_jasa_opcf = df_jasa_opcf.groupby(["JASA"]).sum()
        
        other_opcf = []
        for column_name, item in df_jasa_opcf.items():
            temp = item["Subtotal"]
            # 引く項目のリスト
            items_to_subtract = [
                "Net Income Before Taxes",
                "Depreciation/Depletion",
                "Amortization of Goodwill",
                "Depreciation and Amortization on Other",
                "Increase (Decrease) in Provision",
                "Interest and Dividend Income CF",
                "Interest Expenses CF",
                "Other Loss (Gain)",
                "Decrease (Increase) in Trade Receivables",
                "Decrease (Increase) in Inventories",
                "Increase (Decrease) in Advances Received",
                "Decrease (Increase) in Prepaid Expenses",
                "Increase (Decrease) in Trade Payables",
                "Increase (Decrease) in Accounts Payable - Other, and Accrued Expenses"
            ]
            for key in items_to_subtract:
                temp -= item[key]
            other_opcf.append(temp)
        
        df_jasa_opcf.loc["Other, Net"] = other_opcf

        other_pr = []
        for column_name, item in df_jasa_opcf.items():
            temp = item["Cash from Operating Activities"]
            # 引く項目のリスト
            items_to_subtract = [
                "Subtotal",
                "Interest and Dividends Received",
                "Interest Paid",
                "Proceeds from Insurance Income",
                "Compensation Paid for Damage",
                "Cash Taxes Paid"
            ]
            for key in items_to_subtract:
                temp -= item[key]
            other_pr.append(temp)

        df_jasa_opcf.loc["Other Cash from Operating Activities"] = other_pr

        df_jasa_opcf = df_jasa_opcf.rename(index={"Net Income Before Taxes":"Net Income/Starting Line"})

        return df_jasa_opcf

    # 営業CFの直接法ver.
    def JASA_opcf2(df_info, df_key_list, filtered_df):
        jasa_opcf2_code = ["Receipts from Operating Revenue",
                           "Payments for Raw Materials and Goods",
                           "Payments of Personnel Expenses",
                           "Subtotal",
                           "Interest and Dividends Received",
                            "Interest Paid",
                            "Proceeds from Insurance Income",
                            "Compensation Paid for Damage",
                            "Cash Taxes Paid",
                            "Cash from Operating Activities"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") 
        join_df = join_df.fillna(0.1)
        df_jasa_opcf2 = join_df[join_df["JASA"].isin(jasa_opcf2_code)]
        df_jasa_opcf2 = df_jasa_opcf2.drop("要素ID", axis = 1)
        df_jasa_opcf2.iloc[:, 1:] = df_jasa_opcf2.iloc[:, 1:].astype(np.float64)
        df_jasa_opcf2 = df_jasa_opcf2.groupby(["JASA"]).sum()
        
        other_opcf2 = []
        for column_name, item in df_jasa_opcf2.items():
            temp = item["Subtotal"]
            # 引く項目のリスト
            items_to_subtract = [
                "Receipts from Operating Revenue",
                "Payments for Raw Materials and Goods",
                "Payments of Personnel Expenses"
            ]
            for key in items_to_subtract:
                temp -= item[key]
            other_opcf2.append(temp)
        
        df_jasa_opcf2.loc["Other, Net"] = other_opcf2

        other_pr2 = []
        for column_name, item in df_jasa_opcf2.items():
            temp = item["Cash from Operating Activities"]
            # 引く項目のリスト
            items_to_subtract = [
                "Subtotal",
                "Interest and Dividends Received",
                "Interest Paid",
                "Proceeds from Insurance Income",
                "Compensation Paid for Damage",
                "Cash Taxes Paid"
            ]
            for key in items_to_subtract:
                temp -= item[key]
            other_pr2.append(temp)

        df_jasa_opcf2.loc["Other Cash from Operating Activities"] = other_pr2

        return df_jasa_opcf2

    # 投資CF
    def JASA_incf_NonConsolidated(df_info, df_key_list, filtered_df):
        jasa_incf_code = ["Purchase of Securities",
                          "Proceeds from Sale of Securities",
                          "Purchase of Property, Plant and Equipment",
                          "Proceeds from Sale of Property, Plant and Equipment",
                          "Purchase of Intangible Assets",
                          "Proceeds from Sale of Intangible Assets",
                          "Purchase of Investment Securities",
                          "Proceeds from Sale of Investment Securities",
                          "Loans Advances",
                          "Proceeds from Collection of Loans Receivable",
                          "Payments into Time Deposits",
                          "Proceeds from Withdrawal of Time Deposits",
                          "Payments of Leasehold and Guarantee Deposits",
                          "Proceeds from Refund of Leasehold and Guarantee Deposits",
                          "Payments for Acquisition of Businesses",
                          "Proceeds from Sale of Businesses",
                          "Cash from Investing Activities"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") 
        join_df = join_df.fillna(0.1)
        df_jasa_incf = join_df[join_df["JASA"].isin(jasa_incf_code)]
        df_jasa_incf = df_jasa_incf.drop("要素ID", axis = 1)
        df_jasa_incf.iloc[:, 1:] = df_jasa_incf.iloc[:, 1:].astype(np.float64)
        df_jasa_incf = df_jasa_incf.groupby(["JASA"]).sum()

        other_incf = []
        for column_name, item in df_jasa_incf.items():
            temp = item["Cash from Investing Activities"]
            # 引く項目のリスト
            items_to_subtract = ["Purchase of Securities",
                "Proceeds from Sale of Securities",
                "Purchase of Property, Plant and Equipment",
                "Proceeds from Sale of Property, Plant and Equipment",
                "Purchase of Intangible Assets",
                "Proceeds from Sale of Intangible Assets",
                "Purchase of Investment Securities",
                "Proceeds from Sale of Investment Securities",
                "Loans Advances",
                "Proceeds from Collection of Loans Receivable",
                "Payments into Time Deposits",
                "Proceeds from Withdrawal of Time Deposits",
                "Payments of Leasehold and Guarantee Deposits",
                "Proceeds from Refund of Leasehold and Guarantee Deposits",
                "Payments for Acquisition of Businesses",
                "Proceeds from Sale of Businesses"]
            for key in items_to_subtract:
                temp -= item[key]
            other_incf.append(temp)
        
        df_jasa_incf.loc["Other Cash from Investing Activities"] = other_incf

        return df_jasa_incf

    # 財務CF
    def JASA_ficf_NonConsolidated(df_info, df_key_list, filtered_df):
        jasa_ficf_code= [ "Interest Paid",
                          "Proceeds from Short Term Borrowings",
                          "Repayments of Short Term Borrowings",
                          "Proceeds from Long Term Borrowings",
                          "Repayments of Long Term Borrowings",
                          "Proceeds from Issuance of Bonds",
                          "Redemption of Bonds",
                          "Proceeds from Issuance of Shares",
                          "Purchase of Treasury Shares",
                          "Proceeds from Sale of Treasury Shares",
                          "Repayments of Lease Liabilities",
                          "Dividends Paid",
                          "Cash from Financing Activities",
                          "Foreign Exchange Effects",
                          "Net Change in Cash",
                          "Ending Cash Balance"
                          ]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") 
        join_df = join_df.fillna(0.1)
        df_jasa_ficf = join_df[join_df["JASA"].isin(jasa_ficf_code)]
        df_jasa_ficf = df_jasa_ficf.drop("要素ID", axis = 1)
        df_jasa_ficf.iloc[:, 1:] = df_jasa_ficf.iloc[:, 1:].astype(np.float64)
        df_jasa_ficf = df_jasa_ficf.groupby(["JASA"]).sum()

        other_ficf = []
        for column_name, item in df_jasa_ficf.items():
            temp = item["Cash from Financing Activities"]
            # 引く項目のリスト
            items_to_subtract = [
                "Interest Paid",
               "Proceeds from Short Term Borrowings",
                "Repayments of Short Term Borrowings",
                "Proceeds from Long Term Borrowings",
                "Repayments of Long Term Borrowings",
                "Proceeds from Issuance of Bonds",
                "Redemption of Bonds",
                "Purchase of Treasury Shares",
                "Proceeds from Sale of Treasury Shares",
                "Repayments of Lease Liabilities",
                "Dividends Paid"]            
            for key in items_to_subtract:
                temp -= item[key]
            other_ficf.append(temp)
        
        df_jasa_ficf.loc["Other Cash from Financing Activities"] = other_ficf

        return df_jasa_ficf


    def JASA_date(df_info, df_key_list, filtered_df):
        jasa_date_code = ["jpdei_cor:CurrentPeriodEndDateDEI"]
        join_df = pd.merge(df_key_list, df_info, on = "要素ID", how = "outer")
        #join_df = join_df.dropna(subset = "JASA") #ここを消した場合、別記事業codeが入って、error(float + str)が出るはず。でもここが問題か？
        join_df = join_df.fillna(0)
        df_jasa_date = join_df[join_df["要素ID"].isin(jasa_date_code)]
        df_jasa_date = df_jasa_date.drop_duplicates(subset="要素ID")
        df_jasa_date = df_jasa_date.drop("要素ID", axis = 1)
        #df_jasa_date.iloc[:, 1:] = df_jasa_date.iloc[:, 1:].astype(np.float64)
        #df_jasa_equity = df_jasa_equity.groupby(["JASA"]).sum()

        return df_jasa_date
    
    df_jasa_date = JASA_date(df_info, df_key_list, filtered_df)
    df_jasa_ca = JASA_ca(df_info, df_key_list, filtered_df)
    df_jasa_nca = JASA_nca(df_info, df_key_list, filtered_df)
    df_jasa_cl = JASA_cl(df_info, df_key_list, filtered_df)
    df_jasa_ncl = JASA_ncl(df_info, df_key_list, filtered_df)
    df_jasa_equity = JASA_equity_NonConsolidated(df_info, df_key_list, filtered_df)
    df_jasa_revenue = JASA_revenue(df_info, df_key_list, filtered_df)
    df_jasa_pl = JASA_pl_NonConsolidated(df_info, df_key_list, filtered_df)
    df_jasa_opcf = JASA_opcf_NonConsolidated(df_info, df_key_list, filtered_df)
    df_jasa_opcf2 = JASA_opcf2(df_info, df_key_list, filtered_df)
    df_jasa_incf = JASA_incf_NonConsolidated(df_info, df_key_list, filtered_df)
    df_jasa_ficf = JASA_ficf_NonConsolidated(df_info, df_key_list, filtered_df)

    df1 = df_jasa_ca
    df2 = pd.concat([df1, df_jasa_nca])
    df3 = pd.concat([df2, df_jasa_cl])
    df4 = pd.concat([df3, df_jasa_ncl])
    df5 = pd.concat([df4, df_jasa_equity])
    df6 = pd.concat([df5, df_jasa_revenue])
    df7 = pd.concat([df6, df_jasa_pl])
    df8 = pd.concat([df7, df_jasa_opcf])
    df9 = pd.concat([df8, df_jasa_incf])
    df10 = pd.concat([df9, df_jasa_ficf])
    df10 = df10[~df10.index.duplicated(keep='first')]  # 最初に登場するもの以外の重複を削除

    df_jasa = df10.reindex(index=["Total Current Assets", 
                                "Cash and Short Term Investments",
                                "Notes Receivable/Accounts Receivable True",
                                "Total Inventory True",
                                "Prepaid Expenses",
                                "Trading Securities",
                                "Other Current Assets",
                                "Total Non-Current Assets",
                                "Property/Plant/Equipment, Total - Net",
                                "Goodwill, Net",
                                "Other Intangibles, Net",
                                "Long Term Investments",
                                "Leasehold and Guarantee Deposits True",
                                "Deferred Tax Assets",
                                "Other Non-Current Assets",
                                "Total Assets",
                                "Total Current Liabilities",
                                "Notes Payable/Accounts Payable True",
                                "Advances Received True",
                                "Capital Leases",
                                "Short Term Debt",
                                "Current Port. of LT Debt",
                                "Other Current Liabilities",
                                "Total Non-Current Liabilities",
                                "Long Term Debt",
                                "Bonds",
                                "Lease Liabilities",
                                "Asset Retirement Obligation",
                                "Retirement Benefit",
                                "Deferred Tax Liabilities",
                                "Other Non-Current Liabilities",
                                "Total Liabilities",
                                "Common Stock",
                                "Total, Additional Paid-In",
                                "Retained Earnings",
                                "Treasury Stock-Common",
                                "Valuation/Exchange Differences, etc.",
                                "Other Equity, Total",
                                "Net Assets",
                                "Total Liabilities & Net Assets",
                                " ",
                                "Total Revenue True",
                                "Cost of Revenue, Total True",
                                "Gross Profit True",
                                "Total Operating Expenses",
                                "Selling/General/Admin. Expenses, Total",
                                "Logistics Costs True",
                                "Personnel Expenses True",
                                "Advertising Expenses",
                                "Outsourcing Cost",
                                "Rents",
                                "Research & Development",
                                "Depreciation / Amortization True",
                                "Other Selling/General/Admin. Expenses",
                                "Operating Income True",
                                "Non-Operating Income",
                                "Interest and Dividend Income PL True",
                                "Gains From Foreign Exchange",
                                "Subsidies",
                                "Other Non-Operating Income",
                                "Non-Operating Expenses",
                                "Interest Expenses PL True",
                                "Loss From Foreign Exchange",
                                "Transaction Fees",
                                "Other Non-Operating Expenses",
                                "Ordinary Profit",
                                "Extraordinary Income",
                                "Extraordinary Loss",
                                "Net Income Before Taxes",
                                "Provision for Income Taxes",
                                "Net Income",
                                " ",
                                "Net Income/Starting Line",
                                "Depreciation/Depletion",
                                "Amortization of Goodwill",
                                "Depreciation and Amortization on Other",
                                "Increase (Decrease) in Provision",
                                "Interest and Dividend Income CF",
                                "Interest Expenses CF",
                                "Other Loss (Gain)",
                                "Decrease (Increase) in Trade Receivables",
                                "Decrease (Increase) in Inventories",
                                "Increase (Decrease) in Advances Received",
                                "Decrease (Increase) in Prepaid Expenses",
                                "Increase (Decrease) in Trade Payables",
                                "Increase (Decrease) in Accounts Payable - Other, and Accrued Expenses",
                                "Other, Net",
                                "Subtotal",
                                "Interest and Dividends Received",
                                "Interest Paid",
                                "Proceeds from Insurance Income",
                                "Compensation Paid for Damage",
                                "Cash Taxes Paid",
                                "Other Cash from Operating Activities",
                                "Cash from Operating Activities",
                                "Purchase of Securities",
                                "Proceeds from Sale of Securities",
                                "Purchase of Property, Plant and Equipment",
                                "Proceeds from Sale of Property, Plant and Equipment",
                                "Purchase of Intangible Assets",
                                "Proceeds from Sale of Intangible Assets",
                                "Purchase of Investment Securities",
                                "Proceeds from Sale of Investment Securities",
                                "Loans Advances",
                                "Proceeds from Collection of Loans Receivable",
                                "Payments into Time Deposits",
                                "Proceeds from Withdrawal of Time Deposits",
                                "Payments of Leasehold and Guarantee Deposits",
                                "Proceeds from Refund of Leasehold and Guarantee Deposits",
                                "Payments for Acquisition of Businesses",
                                "Proceeds from Sale of Businesses",
                                "Other Cash from Investing Activities",
                                "Cash from Investing Activities",
                                "Proceeds from Short Term Borrowings",
                                "Repayments of Short Term Borrowings",
                                "Proceeds from Long Term Borrowings",
                                "Repayments of Long Term Borrowings",
                                "Proceeds from Issuance of Bonds",
                                "Redemption of Bonds",
                                "Proceeds from Issuance of Shares",
                                "Purchase of Treasury Shares",
                                "Proceeds from Sale of Treasury Shares",
                                "Repayments of Lease Liabilities",
                                "Dividends Paid",
                                "Other Cash from Financing Activities",
                                "Cash from Financing Activities",
                                " ",
                                "Foreign Exchange Effects",
                                "Net Change in Cash",
                                "Ending Cash Balance"])

    df_jasa = df_jasa.rename(columns={"Notes Receivable/Accounts Receivable True":"Notes Receivable/Accounts Receivable", 
                                      "Total Inventory True":"Total Inventory",
                                      "Leasehold and Guarantee Deposits True":"Leasehold and Guarantee Deposits",
                                      "Notes Payable/Accounts Payable True":"Notes Payable/Accounts Payable",
                                      "Advances Received True":"Advances Received",
                                      "Total Revenue True":"Total Revenue",
                                      "Cost of Revenue, Total True":"Cost of Revenue, Total",
                                      "Gross Profit True":"Gross Profit",
                                      "Logistics Costs True":"Logistics Costs",
                                      "Personnel Expenses True":"Personnel Expenses",
                                      "Depreciation / Amortization True":"Depreciation / Amortization",
                                      "Operating Income True":"Operating Income",
                                      "Interest Expenses PL True":"Interest Expenses PL"})

    # 0より大きく10より小さい値を欠損値に置換
    df_jasa = df_jasa.apply(lambda col: col.map(lambda x: np.nan if (x > 0 and x < 10) else x))

    # 全てのセルの値を下3桁が000になるように変換
    df_jasa = df_jasa.apply(lambda col: col.map(lambda x: np.floor(x / 1000) * 1000 if isinstance(x, (int, float)) else x))

    #df_jasa = df_jasa.drop("Unnamed: 0", axis = 1)

    #欠損値以外をint型に変換
    df_jasa = df_jasa.fillna(-1).round().astype(np.float64)
    df_jasa = df_jasa.replace(-1, "N/A")

    return df_jasa







