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

    class table_sample_make_v2
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
        /// 读html路径，写待标注的表格样本
        /// </summary>
        public static void GetTabelSample()
        {
            string[] paths = File.ReadAllLines(@"表格样本.txt");
            string TablePath = @"样例-Pseudo-0119.json";

            // 超参指定模型行列
            //int row_max = 10;
            //int col_max = 10;
            //V2版本下 长宽由模型指定，C#不做限制

            List<Label_Studio> reuslts = new List<Label_Studio>();
            try
            {
                for (int index_path = 0; index_path < 20; index_path++)
                {
                    string contents = File.ReadAllText(paths[index_path]);

                    //contents = File.ReadAllText(@"\\fileserver.finchina.local\解析附件\解析文件新\火车头专用\债券公告\上交所私募债公告\2023\202305\2023-05-29\251180-泸州临港投资集团有限公司2023年面向专业投资者非公开发行公司债券（第一期）募集说明书-251180_20230529_1GHX.html");

                    if (contents.Length == 0) continue;

                    contents = Regex.Replace(contents, @"\p{C}", "");
                    contents = Regex.Replace(contents, @"\s+", " ");
                    contents = html_mark(contents);

                    // 激进的策略 by YWJ 因为label studio 不接受连续的空格
                    contents = Regex.Replace(contents, @"\s+", "");

                    contents = Parse.Common.Util.ToDBC(contents);

                    List<Match> matches = Regex.Matches(contents, @"<table[^<>]*>[\s\S]*?</table>").Cast<Match>().ToList();

                    for (int index_table = 0; index_table < matches.Count; index_table++)  // 遍历每一个表格
                    {
                        // 对单元格进行定位
                        string this_table = matches[index_table].Value;
                        List<Match> tds = Regex.Matches(this_table, @"(?<=<td[^<>]*>)[\s\S]*?(?=</td>)").Cast<Match>().ToList();
                        if (tds.Count == 0)
                        {
                            continue;
                        }
                        for (int i = tds.Count - 1; i >= 0; i--)
                        {
                            string new_content;
                            if (string.IsNullOrWhiteSpace(tds[i].Value)) new_content = string.Format(@"{1}@{0}", i + 1, "空白");
                            else if (tds[i].Value == "-") new_content = string.Format(@"{1}@{0}", i + 1, "－");
                            else new_content = string.Format(@"{1}@{0}", i + 1, tds[i].Value.Trim());
                            // 对指的位置做剔除和插入
                            this_table = this_table.Remove(tds[i].Index, tds[i].Length);
                            this_table = this_table.Insert(tds[i].Index, new_content);
                        }

                        // 通过iTabel对表格进行行列拓展，已知的问题有如下3个
                        // 1、遇到 th 标签会卡死
                        // 2、单元格内部有 < > 会替换为 &lt; &gt;
                        // 3、单元格内部只有 - 会替换为 空
                        this_table = Regex.Replace(this_table, @"<th[^<>]*>", "");
                        this_table = Regex.Replace(this_table, @"</th>", "");

                        // 下面的操作主要是要获得行列拓展前的表格str和行列拓展后的表格str
                        string table_str_pre = RemoveExtraAttributes(this_table);
                        string table_str_fit = GetTable(table_str_pre, config);
                        table_str_fit = Regex.Replace(table_str_fit, @"\s+", ""); // iTabel会带来空格

                        // 做成好看的表格样式，详细参数咨询www.baidu.com
                        string one_table_new = table_str_pre.Replace("<table>", @"<table border=""1"" cellpadding=""0"" cellspacing=""0"" height=""600px"">");

                        one_table_new += "<br/>";
                        one_table_new += "<p>" + paths[index_path] + "<p/>";
                        one_table_new += string.Format(@"<p>表格{0}</p>", index_table + 1);
                        one_table_new += "<p>" + "以下是行列拓展后表格，业务无需标注，供参考使用" + "<p/>";
                        one_table_new += "<br/>";
                        one_table_new += table_str_fit.Replace("<table>", @"<table border=""1"" cellpadding=""0"" cellspacing=""0"">");

                        string one_html = CreateHtml(one_table_new);


                        // 下面开始造输入格式样本
                        Label_Studio _Studio = new Label_Studio
                        {
                            data = new Dictionary<string, string>(),
                            //annotations = new List<Label_Studio_2> { new Label_Studio_2() },
                            //predictions = new List<Label_Studio_2> { new Label_Studio_2() },
                        };
                        _Studio.data.Add("html", one_html);
                        bool do_you_have_model = true; // 没训练好的模型就止步于此吧
                        if (!do_you_have_model) goto end;

                        //_Studio.annotations = new List<Label_Studio_2> { new Label_Studio_2() };
                        _Studio.predictions = new List<Label_Studio_2> { new Label_Studio_2() };
                        //_Studio.annotations[0].result = new List<object>();
                        _Studio.predictions[0].result = new List<object>();
                        




                        #region 如果有训练好的模型，可以用以下的代码打伪标签
                        List<List<object>> model_result = JsonConvert.DeserializeObject<List<List<object>>>(Http(table_str_fit, "http://10.17.107.66:10086/prediction", 100000));


                        List<Result> results = new List<Result>();
                        foreach (var itemm in model_result)
                        {
                            string label = itemm[0].ToString();
                            int row_number = Convert.ToInt32(itemm[1]);
                            int col_number = Convert.ToInt32(itemm[2]);
                            decimal confidence = Convert.ToDecimal(itemm[3]);
                            string text = itemm[4].ToString();
                            int entity_ids = Convert.ToInt32(itemm[5]);
                            results.Add(new Result
                            {
                                label = label,
                                row_number = row_number,
                                col_number = col_number,
                                entity_ids = entity_ids,
                                confidence = confidence,
                                text = text,
                            });
                        }

                        // 打伪标签时，需要打到原表格上，此时需用iTabel操作一次，获取entity_ids对应的row_number col_number
                        List<Tuple<int, int, int>> locations = GetOriginalLocation(table_str_pre);

                        foreach (var itemm in results.GroupBy(i => i.entity_ids))
                        {
                            // 此处选了置信度最高的 多标签分类的话，这里要改改 取list
                            Result result = itemm.OrderByDescending(i => i.confidence).First();

                            string label = result.label;
                            int row_number = result.row_number;
                            int col_number = result.col_number;
                            decimal confidence = result.confidence;

                            // 根据索引去寻找位置，找不到就不要了
                            var one_location = locations.FirstOrDefault(it => it.Item1 == result.entity_ids);
                            if (one_location == null) continue;
                            Label_Studio_3 _Studio_3 = new Label_Studio_3
                            {
                                value = new Label_Studio_4
                                {
                                    start = string.Format("/table[1]/tbody[1]/tr[{0}]/td[{1}]/text()[1]", one_location.Item2, one_location.Item3),
                                    end = string.Format("/table[1]/tbody[1]/tr[{0}]/td[{1}]/text()[1]", one_location.Item2, one_location.Item3),
                                    startOffset = 0,
                                    endOffset = (result.text + "@" + result.entity_ids).Length,
                                    text = result.text + "@" + result.entity_ids,
                                    labels = new List<string> { label }
                                },
                                from_name = "ner",
                                to_name = "text",
                                type = "labels"
                            };
                            _Studio.predictions[0].result.Add(_Studio_3);
                        }
                    #endregion

                    end:
                        reuslts.Add(_Studio);
                    }
                }
            }
            catch(Exception e)
            {

            }
            var settings = new JsonSerializerSettings
            {
                NullValueHandling = NullValueHandling.Ignore
            };
            File.WriteAllText(TablePath, JsonConvert.SerializeObject(reuslts, settings));
        }
        /// <summary>
        /// 获取原始的索引和定位（获取entity_ids对应的row_number col_number）
        /// </summary>
        /// <param name="table"></param>
        /// <param name="locations"></param>
        private static List<Tuple<int, int, int>> GetOriginalLocation(string table)
        {
            List<Tuple<int, int, int>> locations = new List<Tuple<int, int, int>>();
            List<string> rows = Regex.Matches(table, @"<tr[^<>]*>[\s\S]*?</tr>").Cast<Match>().Select(i => i.Value).ToList();

            for (int i = 0; i < rows.Count; i++)
            {
                List<string> cols = Regex.Matches(rows[i], @"(?<=<td[^<>]*>)[\s\S]*?(?=</td>)").Cast<Match>().Select(ii => ii.Value).ToList();

                for (int j = 0; j < cols.Count; j++)
                {
                    // 匹配位置
                    Match match = Regex.Match(cols[j], @"(.*?)@(\d+)$");
                    if (match.Success)
                    {
                        locations.Add(new Tuple<int, int, int>(Convert.ToInt32(match.Groups[2].Value), i+1, j+1));
                    }

                }
            }
            return locations;
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
        private static string CreateHtml(string content)
        {
            //StringBuilder sw = new StringBuilder();
            ////sw.AppendLine("<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">");
            //sw.AppendLine("<html xmlns=\"http://www.w3.org/1999/xhtml\">");
            //sw.AppendLine("<head>");
            //sw.AppendLine("<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"/>");
            //sw.AppendLine("<title> </title>");
            //sw.AppendLine("</head>");
            //sw.AppendLine("<body style=\"word-wrap: break-word; overflow-wrap: break-word; overflow: hidden \">");
            //sw.AppendLine(content);
            //sw.AppendLine("</body>");
            //sw.AppendLine("</html>");

            return "<body style=\"word-wrap: break-word; overflow-wrap: break-word; overflow: hidden \">" + content + "</body>";

            //return sw.ToString();
        }
        /// <summary>
        /// 剔除除了 rowspan 和 colspan 以外的所有属性
        /// </summary>
        /// <param name="html"></param>
        /// <returns></returns>
        static string RemoveExtraAttributes(string html)
        {
            // 实现方式是先捕获所有的tr 再遍历每一个tr，捕获所有的td，再用正则匹配当前的rowspan和colspan，再自己拼属性回去，正则示例如下
            // (?<=rowspan\s*=\s*(?'quto'""|')?)\d+(?=(?:\k'quto')?)
            // (?<=colspan\s*=\s*(?'quto'""|')?)\d+(?=(?:\k'quto')?)
            // var colSpan = match.Success ? Convert.ToInt32(cm.Value) : Cell.DefaultSpan;
            // $"<td rowspan=\"{RowSpan}\" colspan=\"ColSpan}\">{Value}</td>"

            // 匹配tr标签的正则表达式
            string trRegex = @"<tr[^<>]*>[\s\S]*?</tr>";

            // 匹配td标签的正则表达式
            string tdRegex = @"<td[^<>]*>([\s\S]*?)</td>";

            // 匹配rowspan属性的正则表达式
            string rowspanRegex = @"(?<=rowspan\s*=\s*(?'quto'""|')?)\d+(?=(?:\k'quto')?)";

            // 匹配colspan属性的正则表达式
            string colspanRegex = @"(?<=colspan\s*=\s*(?'quto'""|')?)\d+(?=(?:\k'quto')?)";

            Regex trTagRegex = new Regex(trRegex);
            Regex tdTagRegex = new Regex(tdRegex);
            Regex rowspanAttrRegex = new Regex(rowspanRegex);
            Regex colspanAttrRegex = new Regex(colspanRegex);

            List<string> results = new List<string>();

            // 匹配所有tr标签
            MatchCollection trMatches = trTagRegex.Matches(html);
            results.Add("<table>");
            // 遍历每一个tr标签
            foreach (Match trMatch in trMatches)
            {
                results.Add("<tr>");
                string trHtml = trMatch.Value;

                // 匹配所有td标签
                MatchCollection tdMatches = tdTagRegex.Matches(trHtml);

                // 遍历每一个td标签
                foreach (Match tdMatch in tdMatches)
                {
                    string tdHtml = tdMatch.Value;

                    // 获取rowspan和colspan属性的值
                    Match rowspanMatch = rowspanAttrRegex.Match(tdHtml);
                    Match colspanMatch = colspanAttrRegex.Match(tdHtml);

                    int rowspan = rowspanMatch.Success ? Convert.ToInt32(rowspanMatch.Value) : 1;
                    int colspan = colspanMatch.Success ? Convert.ToInt32(colspanMatch.Value) : 1;

                    // 构造新的td标签，只保留rowspan和colspan属性
                    string newTdHtml = $"<td rowspan=\"{rowspan}\" colspan=\"{colspan}\">{tdMatch.Groups[1].Value}</td>";

                    // 替换原来的td标签-之前加了索引，所以理论上现在没有重复
                    //trHtml = trHtml.Replace(tdHtml, newTdHtml);
                    results.Add(newTdHtml);
                }

                // 替换原来的tr标签
                //html = html.Replace(trMatch.Value, trHtml);
                results.Add("</tr>");
            }
            results.Add("</table>");
            return string.Join("", results);
        }
        /// <summary>
        /// 通过iTable 获取标准表格
        /// iTable封装太严密，后面计划修改，通过HtmlParse做这个事情
        /// </summary>
        /// <param name="table"></param>
        /// <param name="config"></param>
        /// <returns></returns>
        private static string GetTable(string table, string config)
        {
            string getHtml = null;
            string msg = string.Empty;
            Table tb = iTable.HtmlParseHelper.ParseHtmlToTables(table, config, out msg).FirstOrDefault();
            if (msg.Length > 0 || tb == null)
            {
                getHtml = null;
            }
            else
            {
                getHtml = tb.ToTag();
            }
            return getHtml;
        }
        public static string html_mark(string htmlStr)
        {
            htmlStr = htmlStr.Replace(@"�`", "");
            htmlStr = htmlStr.Replace(@"�", "");
            htmlStr = htmlStr.Replace(@"：", ":");
            return Pre_text_and(htmlStr);
        }

        // &类标签的处理
        private static string Pre_text_and(string text)
        {
            if (string.IsNullOrEmpty(text))
                return "";
            else if (!text.Contains("&"))
                return text;

            StringBuilder text_anhao = new StringBuilder(text, text.Length * 2);

            text_anhao.Replace("&#8226;", "·");
            text_anhao.Replace("&mdash;", "-");
            text_anhao.Replace("&lt;", "<");
            text_anhao.Replace("&gt;", ">");
            text_anhao.Replace("&amp;", "&");
            text_anhao.Replace("&ldquo;", "“");
            text_anhao.Replace("&rdquo;", "”");
            text_anhao.Replace("&bull;", "·");
            text_anhao.Replace("&middot;", "·");
            text_anhao.Replace("&permil;", "‰");
            text_anhao.Replace("&le;", "≤");
            text_anhao.Replace("&ge;", "≥");
            text_anhao.Replace("&middot;", "·");
            text_anhao.Replace("&mdash;", "-");
            text_anhao.Replace("&ndash;", "-");
            text_anhao.Replace("&lsquo;", "‘");
            text_anhao.Replace("&rsquo;", "’");
            text_anhao.Replace("&radic;", "√");







            text_anhao.Replace("&shy;", " ");
            text_anhao.Replace("&nbsp;", " ");
            text_anhao.Replace("&ensp;", " ");
            text_anhao.Replace("&emsp;", " ");
            text_anhao.Replace("&thinsp;", " ");
            text_anhao.Replace("&zwnj;", " ");
            text_anhao.Replace("&zwj;", " ");
            text_anhao.Replace("&#x0020;", " ");
            text_anhao.Replace("&#x0009;", " ");
            text_anhao.Replace("&#x000D;", " ");
            text_anhao.Replace("&#x000A;", " ");
            text_anhao.Replace("&#12288;", " ");
            text_anhao.Replace("&#xa0;", " ");


            text_anhao.Replace("&Oslash;", "");
            text_anhao.Replace("&not;", "");
            text_anhao.Replace("&#61548;", " ");
            text_anhao.Replace("&#32;", " ");


            text_anhao.Replace("&#33;", "!");
            text_anhao.Replace("&#34;", "\"");
            text_anhao.Replace("&#35;", "#");
            text_anhao.Replace("&#36;", "$");
            text_anhao.Replace("&#37;", "%");
            text_anhao.Replace("&#38;", "&");
            text_anhao.Replace("&#39;", "'");
            text_anhao.Replace("&#40;", "(");
            text_anhao.Replace("&#41;", ")");
            text_anhao.Replace("&#42;", "*");
            text_anhao.Replace("&#43;", "+");
            text_anhao.Replace("&#44;", ",");
            text_anhao.Replace("&#45;", "-");
            text_anhao.Replace("&#46;", ".");
            text_anhao.Replace("&#47;", "/");
            text_anhao.Replace("&#48;", "0");
            text_anhao.Replace("&#49;", "1");
            text_anhao.Replace("&#50;", "2");
            text_anhao.Replace("&#51;", "3");
            text_anhao.Replace("&#52;", "4");
            text_anhao.Replace("&#53;", "5");
            text_anhao.Replace("&#54;", "6");
            text_anhao.Replace("&#55;", "7");
            text_anhao.Replace("&#56;", "8");
            text_anhao.Replace("&#57;", "9");
            text_anhao.Replace("&#58;", ":");
            text_anhao.Replace("&#59;", ";");
            text_anhao.Replace("&#60;", "<");
            text_anhao.Replace("&#61;", "=");
            text_anhao.Replace("&#62;", ">");
            text_anhao.Replace("&#63;", "?");
            text_anhao.Replace("&#64;", "@");
            text_anhao.Replace("&#65;", "A");
            text_anhao.Replace("&#66;", "B");
            text_anhao.Replace("&#67;", "C");
            text_anhao.Replace("&#68;", "D");
            text_anhao.Replace("&#69;", "E");
            text_anhao.Replace("&#70;", "F");
            text_anhao.Replace("&#71;", "G");
            text_anhao.Replace("&#72;", "H");
            text_anhao.Replace("&#73;", "I");
            text_anhao.Replace("&#74;", "J");
            text_anhao.Replace("&#75;", "K");
            text_anhao.Replace("&#76;", "L");
            text_anhao.Replace("&#77;", "M");
            text_anhao.Replace("&#78;", "N");
            text_anhao.Replace("&#79;", "O");
            text_anhao.Replace("&#80;", "P");
            text_anhao.Replace("&#81;", "Q");
            text_anhao.Replace("&#82;", "R");
            text_anhao.Replace("&#83;", "S");
            text_anhao.Replace("&#84;", "T");
            text_anhao.Replace("&#85;", "U");
            text_anhao.Replace("&#86;", "V");
            text_anhao.Replace("&#87;", "W");
            text_anhao.Replace("&#88;", "X");
            text_anhao.Replace("&#89;", "Y");
            text_anhao.Replace("&#90;", "Z");
            text_anhao.Replace("&#91;", "[");
            text_anhao.Replace("&#92;", "\\");
            text_anhao.Replace("&#93;", "]");
            text_anhao.Replace("&#94;", "^");
            text_anhao.Replace("&#95;", "_");
            text_anhao.Replace("&#96;", "`");
            text_anhao.Replace("&#97;", "a");
            text_anhao.Replace("&#98;", "b");
            text_anhao.Replace("&#99;", "c");
            text_anhao.Replace("&#100;", "d");
            text_anhao.Replace("&#101;", "e");
            text_anhao.Replace("&#102;", "f");
            text_anhao.Replace("&#103;", "g");
            text_anhao.Replace("&#104;", "h");
            text_anhao.Replace("&#105;", "i");
            text_anhao.Replace("&#106;", "j");
            text_anhao.Replace("&#107;", "k");
            text_anhao.Replace("&#108;", "l");
            text_anhao.Replace("&#109;", "m");
            text_anhao.Replace("&#110;", "n");
            text_anhao.Replace("&#111;", "o");
            text_anhao.Replace("&#112;", "p");
            text_anhao.Replace("&#113;", "q");
            text_anhao.Replace("&#114;", "r");
            text_anhao.Replace("&#115;", "s");
            text_anhao.Replace("&#116;", "t");
            text_anhao.Replace("&#117;", "u");
            text_anhao.Replace("&#118;", "v");
            text_anhao.Replace("&#119;", "w");
            text_anhao.Replace("&#120;", "x");
            text_anhao.Replace("&#121;", "y");
            text_anhao.Replace("&#122;", "z");
            text_anhao.Replace("&#123;", "{");
            text_anhao.Replace("&#124;", "|");
            text_anhao.Replace("&#125;", "}");
            text_anhao.Replace("&#126;", "~");
            text_anhao.Replace("&#255;", "ÿ");

            text_anhao.Replace("&#718;", "ˎ");
            text_anhao.Replace("&#805;", " ");
            text_anhao.Replace("&#8212;", "—");
            text_anhao.Replace("&#8220;", "“");
            text_anhao.Replace("&#8221;", "”");
            text_anhao.Replace("&#8226;", "·");
            text_anhao.Replace("&#8251;", "※");
            text_anhao.Replace("&#8729;", "·");
            text_anhao.Replace("&#8804;", "≤");

            text_anhao.Replace("&#9312;", "①");
            text_anhao.Replace("&#9313;", "②");
            text_anhao.Replace("&#9314;", "③");
            text_anhao.Replace("&#9315;", "④");
            text_anhao.Replace("&#9316;", "⑤");

            text_anhao.Replace("&times;", "×");
            text_anhao.Replace("&times", "×");
            text_anhao.Replace("&phi;", "φ");
            text_anhao.Replace("&rarr;", "→");
            text_anhao.Replace("&yen;", "¥");
            text_anhao.Replace("&deg;", "°");
            text_anhao.Replace("&sup2;", "²");

            text_anhao.Replace("&#9642;", "▪");
            text_anhao.Replace("&#12539;", "・");

            text = text_anhao.ToString();

            return text;
        }
    }
    public class Label_Studio
    {
        public List<Label_Studio_2> annotations { get; set; }  // 标注结果存annotations
        public Dictionary<string, string> data { get; set; }
        public List<Label_Studio_2> predictions { get; set; }  // 模型反标 或者正则反标的数据存predictions
    }
    public class Label_Studio_2
    {
        public List<object> result { get; set; }  // 可能是Label_Studio_3 也可能是Label_Studio_5
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
        public string text { get; set; }
    }
}
