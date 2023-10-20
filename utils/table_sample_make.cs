using iTable.Entities;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace MyWorkSpace
{
    class table_sample_make
    {
        static string config = @"
<?xml version=""1.0""?>
<ArrayOfOperation xmlns:xsd=""http://www.w3.org/2001/XMLSchema"" xmlns:xsi=""http://www.w3.org/2001/XMLSchema-instance"">
  <Operation>
    <Name>行列扩展</Name>
    <ClassName>iTable.Interfaces.TableSpanPad</ClassName>
    <HasParameter>true</HasParameter>
    <DefaultIndex>0</DefaultIndex>
    <Group>All</Group>
    <SerializedValue>&lt;?xml version=""1.0""?&gt;
&lt;TableSpanPad xmlns:xsd=""http://www.w3.org/2001/XMLSchema"" xmlns:xsi=""http://www.w3.org/2001/XMLSchema-instance""&gt;
  &lt;RowSpanNumber&gt;Value&lt;/RowSpanNumber&gt;
  &lt;ColSpanNumber&gt;Value&lt;/ColSpanNumber&gt;
&lt;/TableSpanPad&gt;</SerializedValue>
  </Operation>
</ArrayOfOperation>";
        static string config_empty = @"
<?xml version=""1.0""?>
<ArrayOfOperation xmlns:xsd=""http://www.w3.org/2001/XMLSchema"" xmlns:xsi=""http://www.w3.org/2001/XMLSchema-instance"">
  <Operation>
    <Name>行列扩展</Name>
    <ClassName>iTable.Interfaces.TableSpanPad</ClassName>
    <HasParameter>true</HasParameter>
    <DefaultIndex>0</DefaultIndex>
    <Group>All</Group>
    <SerializedValue>&lt;?xml version=""1.0""?&gt;
&lt;TableSpanPad xmlns:xsd=""http://www.w3.org/2001/XMLSchema"" xmlns:xsi=""http://www.w3.org/2001/XMLSchema-instance""&gt;
  &lt;RowSpanNumber&gt;Empty&lt;/RowSpanNumber&gt;
  &lt;ColSpanNumber&gt;Empty&lt;/ColSpanNumber&gt;
&lt;/TableSpanPad&gt;</SerializedValue>
  </Operation>
</ArrayOfOperation>";
        /// <summary>
        /// 处理 Html 获取 Table 样本
        /// </summary>
        public static void GetTabelSample()
        {
            // list of htmlfile path
            string[] paths = File.ReadAllLines(@"样例.txt");

            string TablePath = @"样例-Pseudo.json";

            // 超参指定模型行列
            int row_max = 10;
            int col_max = 10;

            List<Label_Studio> reuslts = new List<Label_Studio>();
            for (int index_path = 0; index_path < paths.Length; index_path++)
            {
                Console.WriteLine(index_path);


                string contents = Get(paths[index_path]); // Get请求是因为招投标是桶路径

                if (contents.Length == 0) continue;

                contents = Regex.Replace(contents, @"\p{C}", "");
                contents = Parse.Common.Util.ToDBC(contents);

                List<Match> matches = Regex.Matches(contents, @"<table[^<>]*>[\s\S]*?</table>").Cast<Match>().ToList();

                for (int index_table = 0; index_table < matches.Count; index_table++)  // 遍历每一个表格
                {
                    List<Row> one_table = GetTable(matches[index_table].Value, config);
                    List<Row> one_table_empty = GetTable(matches[index_table].Value, config_empty); // 空值填充

                    if (one_table is null || one_table.Count <= 1) continue;

                    int row_count_value = one_table.Count;  // 原值填充后的行数
                    int col_count_value = one_table.Select(item => item.Count).Max();  // 原值填充后的列数

                    int row_count_empty = one_table_empty.Count;
                    int col_count_empty = one_table_empty.Select(item => item.Count).Max();

                    if (row_count_value != row_count_empty || col_count_value != col_count_empty) throw new Exception("len not right");

                    List<List<string>> table_value = new List<List<string>>();
                    List<List<string>> table_empty = new List<List<string>>();
                    for (int i = 0; i < row_count_value; i++)
                    {
                        table_value.Add(new List<string>(col_count_empty));
                        table_empty.Add(new List<string>(col_count_empty));
                    }
                    for (int index_row = 0; index_row < row_count_value; index_row++)
                    {
                        for (int index_col = 0; index_col < col_count_empty; index_col++)
                        {
                            if (index_col < one_table[index_row].Count)
                            {
                                table_value[index_row].Add(one_table[index_row][index_col].Value.Trim());
                                table_empty[index_row].Add(one_table_empty[index_row][index_col].Value.Trim());
                            }
                            else
                            {
                                table_value[index_row].Add("PAD");
                                table_empty[index_row].Add("PAD");
                            }
                        }
                    }

                    #region 剔除重复行
                    string[] table_value_row_values = table_value.Select(item => string.Join(",", item)).ToArray();

                    for (int i = row_count_value - 1; i > 0; i--)
                    {
                        if (table_value_row_values[i] == table_value_row_values[i - 1])
                        {
                            table_value.RemoveAt(i);
                            table_empty.RemoveAt(i);
                        }
                    }

                    #endregion

                    #region 剔除重复列
                    string[] table_value_col_values = new string[col_count_value];

                    for (int i = 0; i < col_count_value; i++)
                    {
                        table_value_col_values[i] = string.Join(",", table_value.Select(item => item[i]));
                    }

                    for (int i = col_count_value - 1; i > 0; i--)
                    {
                        if (table_value_col_values[i] == table_value_col_values[i - 1])
                        {
                            table_value.ForEach(item => item.RemoveAt(i));
                            table_empty.ForEach(item => item.RemoveAt(i));
                        }
                    }
                    #endregion

                    #region 行截取
                    if (table_value.Count > row_max)
                    {
                        table_value = table_value.Take(row_max).ToList();
                        table_empty = table_empty.Take(row_max).ToList();
                    }
                    #endregion

                    // 这里截取了每一个表格固定的大小，逻辑上全自动下用滑窗更合适，半自动下无所谓，外推性会解决这个问题
                    string[,] value_table = GetStandardSizedTable(table_value, row_max, col_max);
                    string[,] empty_table = GetStandardSizedTable(table_empty, row_max, col_max);

                    // 通过原值填充和空值填充的对比 计算每一个值对应的实际坐标 后面优化，用HtmlParse解决这个问题
                    Tuple<int, int>[,] tuples_location = new Tuple<int, int>[row_max, col_max];
                    for (int i = 0; i < row_max; i++)
                    {
                        for (int j = 0; j < col_max; j++)
                        {
                            if (value_table[i, j] == "PAD")
                            {
                                tuples_location[i, j] = new Tuple<int, int>(i, j);
                                continue;
                            }
                            if (value_table[i, j] == empty_table[i, j])
                            {
                                tuples_location[i, j] = new Tuple<int, int>(i, j);
                                continue;
                            }
                            int this_row = i;
                            int this_col = j;

                            // 优先寻找行合并
                            while (this_col > 0)
                            {
                                if (value_table[this_row, this_col] == value_table[this_row, this_col - 1]) this_col--;
                                else break;
                            }
                            if (this_col == j)
                            {
                                while (this_row > 0)
                                {
                                    if (value_table[this_row, this_col] == value_table[this_row - 1, this_col]) this_row--;
                                    else break;
                                }
                            }
                            tuples_location[i, j] = new Tuple<int, int>(this_row, this_col);
                        }
                    }

                    string table_cutting = GetTables(value_table, tuples_location, true); // 第一个表格截断 带实际位置索引
                    string table_nocutting = GetTables(value_table, null, false); // 第二个表格不截断 不带实际位置索引

                    // 做成好看的表格样式，详细参数咨询www.baidu.com
                    string one_table_new = table_cutting.Replace("<table>", @"<table border=""1"" cellpadding=""0"" cellspacing=""0"" height=""600px"">");

                    one_table_new += "<br/>";
                    one_table_new += "<p>" + paths[index_path] + "<p/>";
                    one_table_new += string.Format(@"<p>表格{0}</p>", index_table + 1);
                    one_table_new += "<br/>";

                    one_table_new += table_nocutting.Replace("<table>", @"<table border=""1"" cellpadding=""0"" cellspacing=""0"">");
                    string one_html = CreateHtml(one_table_new);


                    // 下面开始造输入格式样本
                    Label_Studio _Studio = new Label_Studio
                    {
                        data = new Dictionary<string, string>(),
                        annotations = new List<Label_Studio_2> { new Label_Studio_2() },
                    };
                    _Studio.annotations[0].result = new List<object>();
                    _Studio.data.Add("html", one_html);


                    bool do_you_have_model = true;
                    if (!do_you_have_model) goto end;

                    #region 如果有训练好的模型，可以用以下的代码打伪标签
                    List<string> tds = Regex.Matches(table_cutting, @"<td>(.*?)</td>").Cast<Match>().Select(itemm => itemm.Groups[1].Value).ToList();
                    List<string> tds_nocut = Regex.Matches(GetTables(value_table, tuples_location, false), @"<td>(.*?)</td>").Cast<Match>().Select(itemm => itemm.Groups[1].Value).ToList();

                    List<string> guids = new List<string>();
                    for (int i = 0; i < row_max * col_max; i++)
                    {
                        guids.Add(Guid.NewGuid().ToString());
                    }

                    if (tds.Count != row_max * col_max) throw new Exception("count of tds is not right");

                    List<List<object>> model_result = JsonConvert.DeserializeObject<List<List<object>>>(Http(JsonConvert.SerializeObject(tds_nocut), "http://10.17.107.66:10086/prediction", 100000));


                    List<Result> results = new List<Result>();
                    foreach (var itemm in model_result)
                    {
                        string label = itemm[0].ToString();
                        int row_number = Convert.ToInt32(itemm[1]);
                        int col_number = Convert.ToInt32(itemm[2]);
                        decimal confidence = Convert.ToDecimal(itemm[3]);
                        int entity_ids = row_number * col_max + col_number;
                        results.Add(new Result
                        {
                            label = label,
                            row_number = row_number,
                            col_number = col_number,
                            entity_ids = entity_ids,
                            confidence = confidence,
                        });
                    }

                    foreach (var itemm in results.GroupBy(i => i.entity_ids))
                    {
                        // 此处选了置信度最高的 多标签分类的话，这里要改改 取list
                        Result result = itemm.OrderByDescending(i => i.confidence).First();

                        string label = result.label;
                        int row_number = result.row_number;
                        int col_number = result.col_number;
                        decimal confidence = result.confidence;
                        int from_entity_ids = row_number * 10 + col_number;

                        Label_Studio_3 _Studio_3 = new Label_Studio_3
                        {
                            value = new Label_Studio_4
                            {
                                start = string.Format("/table[1]/tbody[1]/tr[{0}]/td[{1}]/text()[1]", row_number + 1, col_number + 1),
                                end = string.Format("/table[1]/tbody[1]/tr[{0}]/td[{1}]/text()[1]", row_number + 1, col_number + 1),
                                startOffset = 0,
                                endOffset = tds[from_entity_ids].Length,
                                text = tds[from_entity_ids],
                                labels = new List<string> { label }
                            },
                            from_name = "label",
                            to_name = "text",
                            type = "labels"
                        };
                        _Studio.annotations[0].result.Add(_Studio_3);
                    }
                #endregion

                end: 
                    reuslts.Add(_Studio);
                }
            }
            File.WriteAllText(TablePath, JsonConvert.SerializeObject(reuslts));
        }
        /// <summary>
        /// Get请求=
        /// </summary>
        /// <param name="url">请求地址</param>
        /// <param name="error">抛出异常</param>
        /// <returns></returns>
        private static string Get(string url)
        {
            string strMsg = string.Empty;
            try
            {
                WebRequest request = WebRequest.Create(url);
                request.ContentType = "application/x-www-form-urlencoded";
                using (WebResponse response = request.GetResponse())
                {
                    using (StreamReader reader = new StreamReader(response.GetResponseStream(), Encoding.GetEncoding("UTF-8")))
                    {
                        strMsg = reader.ReadToEnd();
                    }
                }
            }
            catch (Exception ex)
            {
                throw new Exception(url + "，Get请求异常：" + ex.Message);
            }
            return strMsg;
        }
        /// <summary>
        /// Http请求
        /// </summary>
        /// <param name="content"></param>
        /// <param name="url"></param>
        /// <returns></returns>
        private static string Http(string content, string url, int timeout)
        {
            string strMsg = string.Empty;
            try
            {
                byte[] requestBuffer = Encoding.GetEncoding("UTF-8").GetBytes(content);

                WebRequest request = WebRequest.Create(url);
                request.Method = "POST";
                request.ContentType = "application/x-www-form-urlencoded";
                request.ContentLength = requestBuffer.Length;
                request.Timeout = Math.Max(60000, timeout);
                using (Stream requestStream = request.GetRequestStream())
                {
                    requestStream.Write(requestBuffer, 0, requestBuffer.Length);
                    requestStream.Close();
                }
                using (WebResponse response = request.GetResponse())
                {
                    using (StreamReader reader = new StreamReader(response.GetResponseStream(), Encoding.GetEncoding("UTF-8")))
                    {
                        strMsg = reader.ReadToEnd();
                        reader.Close();
                    }
                }
            }
            catch (Exception ex)
            {
                throw new Exception(url + ",post请求失败" + ",content为：" + content + "\r\n" + ex.Message);
            }
            return strMsg;
        }
        /// <summary>
        /// 通过iTable 获取标准表格
        /// iTable封装太严密，后面计划修改，通过HtmlParse做这个事情
        /// </summary>
        /// <param name="table"></param>
        /// <param name="config"></param>
        /// <returns></returns>
        private static List<Row> GetTable(string table, string config)
        {
            List<Row> getHtml = null;
            string msg = string.Empty;
            Table tb = iTable.HtmlParseHelper.ParseHtmlToTables(table, config, out msg).FirstOrDefault();
            if (msg.Length > 0 || tb == null)
            {
                getHtml = null;
            }
            else
            {
                getHtml = tb.ToList();
            }
            return getHtml;
        }
        /// <summary>
        /// 获取标准尺寸的表格
        /// </summary>
        /// <param name="list"></param>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        private static string[,] GetStandardSizedTable(List<List<string>> list, int row = 10, int col = 10)
        {
            string[,] returns = new string[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    if (i >= list.Count || j >= list[i].Count) returns[i, j] = "PAD";
                    else if (list[i][j].Length == 0) returns[i, j] = "空白";
                    else returns[i, j] = list[i][j];
                }
            }
            return returns;
        }

        /// <summary>
        /// 获取Xml格式的表格string
        /// </summary>
        /// <param name="one_table"></param>
        /// <param name="location"></param>
        /// <param name="cutting"></param>
        /// <returns></returns>
        private static string GetTables(string[,] one_table, Tuple<int, int>[,] location =null, bool cutting = true)
        {
            string result = "";

            // colmun
            int row_mun = one_table.GetLength(0);
            int col_mun = one_table.GetLength(1);

            result += "<table>";

            for (int i = 0; i < row_mun; i++)
            {
                result += "<tr>";
                for (int j = 0; j < col_mun; j++)
                {
                    string value = one_table[i, j];
                    value = Regex.Replace(value, @"\p{C}", "");
                    if (value != "PAD") value = value.ToLower();
                    value = Parse.Common.Util.ToDBC(value);
                    if (cutting && value.Length > 30) value = value.Substring(0, 30);

                    if(location != null) value += "    " + location[i, j].Item1 + "," + location[i, j].Item2;

                    result += "<td>" + value + "</td>";
                }
                result += "</tr>";
            }
            result += "</table>";

            return result;
        }

        /// <summary>
        /// 没什么好说的
        /// </summary>
        /// <param name="content"></param>
        /// <returns></returns>
        private static string CreateHtml(string content)
        {
            StringBuilder sw = new StringBuilder();
            //sw.AppendLine("<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">");
            sw.AppendLine("<html xmlns=\"http://www.w3.org/1999/xhtml\">");
            sw.AppendLine("<head>");
            sw.AppendLine("<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"/>");
            sw.AppendLine("<title> </title>");
            sw.AppendLine("</head>");
            sw.AppendLine("<body>");
            sw.AppendLine(content);
            sw.AppendLine("</body>");
            sw.AppendLine("</html>");

            return sw.ToString();
        }
    }
    public class Label_Studio
    {
        public List<Label_Studio_2> annotations { get; set; }
        public Dictionary<string, string> data { get; set; }
    }
    public class Label_Studio_2
    {
        public List<object> result { get; set; }
    }
    public class Label_Studio_3 // 实体类
    {
        public string id { get; set; }
        public Label_Studio_4 value { get; set; }
        public string from_name { get; set; } // label 
        public string to_name { get; set; } // text
        public string type { get; set; }  // labels
    }
    public class Label_Studio_5 // 关系类
    {
        public string from_id { get; set; }
        public string to_id { get; set; }
        public string direction { get; set; } // right
        public string type { get; set; }  // relation
    }
    public class Label_Studio_4
    {
        public string start { get; set; }
        public string end { get; set; }
        public int startOffset { get; set; }
        public int endOffset { get; set; }
        public string text { get; set; }
        public List<string> labels { get; set; }
    }
    public class Result
    {
        public string label { get; set; }
        public int row_number { get; set; }
        public int col_number { get; set; }
        public int entity_ids { get; set; }
        public decimal confidence { get; set; }
    }
}
