

class Cleaner:

    def _drop_col_with_one_val(self, df):
        """
        Drop columns with only one
        unique value
        :param df: pandas DataFrame
        :returns pandas DataFrame
        """
        for column in df.columns:
            uniqs = df[column].unique()
            if len(uniqs) == 1:   # using .nunique() doesn't count nans
                df = df.drop(column, axis=1)
        return df

    def _remove_nan_columns(self, df, nan_fraction):
        """
        Remove column in dataframe
        if number of NaN in that
        column is >= `nan_fraction`
        """
        for column in df.columns:
            if df[column].isna().astype(int).mean() >= nan_fraction:
                df = df.drop(column, axis=1)

        return df

    def clean(self, df, nan_fraction):
        """
        Join private functions of the class
        to make all work
        """
        df = self._drop_col_with_one_val(df)
        df = self._remove_nan_columns(df, nan_fraction)
        return df