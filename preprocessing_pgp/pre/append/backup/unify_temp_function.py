
def difference_profile(df1, df2, time_fillna_val, fillna_val, time_cols, cols):
    
    # fillna value to avoid null value when join df
    
    df1 = df1.fillna(fillna_val, subset=cols)
    df2 = df2.fillna(fillna_val, subset=cols)
    
    for c in time_cols:
        df1 = df1.withColumn(c, F.coalesce(F.col(c), F.lit(time_fillna_val)))
        df2 = df2.withColumn(c, F.coalesce(F.col(c), F.lit(time_fillna_val)))
    
    # join df
    difference_df = df1.join(df2, on=df1.columns, how='left_anti')
    
    # revert na value
    for c in time_cols:
        difference_df = difference_df.withColumn(c, F.when(F.col(c) == time_fillna_val, F.lit(None)).otherwise(F.col(c)))
    for c in cols:
        difference_df = difference_df.withColumn(c, F.when(F.col(c) == fillna_val, F.lit(None)).otherwise(F.col(c)))
        
    return difference_df

difference_df = difference_profile(now_profile, yesterday_profile, '1980-01-01', 'no_infor', ['create_date', 'last_active'], ['phone', 'email', 'name'])
difference_df.count()
