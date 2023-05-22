package org.example;

import org.apache.commons.math4.legacy.ml.clustering.CentroidCluster;
import org.apache.commons.math4.legacy.ml.clustering.Clusterable;
import org.apache.commons.math4.legacy.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math4.legacy.stat.descriptive.MultivariateSummaryStatistics;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * アイリスデータセットに対して、KMeans++を使用したクラスタリング
 */
public class Iris {
    public static void main(String[] args) {

        List<IrisRecord> data = getData();

        // 散布図の作成
        generateActualScatterPlotImageSepal(data);
        generateActualScatterPlotImagePetal(data);

        // 統計情報
        MultivariateSummaryStatistics stat = new MultivariateSummaryStatistics(4, false);
        for (IrisRecord row : data) {
            double[] row_data = {row.sepalLength, row.sepalWidth, row.petalLength, row.petalWidth};
            stat.addValue(row_data);
        }

        System.out.println("がく片の長さ(cm)の平均 : " + stat.getMean()[0]);
        // => がく片の長さ(cm)の平均 : 5.843333333333333

        System.out.println("花びらの長さ(cm)の最大と最小 : 最大:" + stat.getMax()[2] + " / 最小:" + stat.getMin()[2]);
        // => 花びらの長さ(cm)の最大と最小 : 最大:6.9 / 最小:1.0

        System.out.println("花びらの幅(cm)の標準偏差 : " + stat.getStandardDeviation()[3]);
        // => 花びらの幅(cm)の標準偏差 : 0.7606126185881713

        System.out.println("属性 : 平均 : 最大 : 最小 : 標準偏差");
        System.out.println("がく片の長さ(cm) : " + stat.getMean()[0] + " : " + stat.getMax()[0] + " : " + stat.getMin()[0] + " : " + stat.getStandardDeviation()[0]);
        System.out.println("がく片の幅(cm) : " + stat.getMean()[1] + " : " + stat.getMax()[1] + " : " + stat.getMin()[1] + " : " + stat.getStandardDeviation()[1]);
        System.out.println("花びらの長さ(cm) : " + stat.getMean()[2] + " : " + stat.getMax()[2] + " : " + stat.getMin()[2] + " : " + stat.getStandardDeviation()[2]);
        System.out.println("花びらの幅(cm) : " + stat.getMean()[3] + " : " + stat.getMax()[3] + " : " + stat.getMin()[3] + " : " + stat.getStandardDeviation()[3]);

        // k-means++法 によるクラスタリング
        List<IrisDataPoint> clusterInput = new ArrayList<>();
        for (IrisRecord record : data) {
            clusterInput.add(new IrisDataPoint(record));
        }
        KMeansPlusPlusClusterer<IrisDataPoint> clusterer = new KMeansPlusPlusClusterer<>(3, 10000);
        List<CentroidCluster<IrisDataPoint>> clusterResults = clusterer.cluster(clusterInput);

        // 分類したクラスターの結果
        for (int i = 0; i < clusterResults.size(); i++) {
            System.out.println("Cluster " + i);

            for (IrisDataPoint point : clusterResults.get(i).getPoints())
                System.out.println(point.getPointInfo());
            System.out.println();
        }

        // 分類したクラスターごとに色分けした散布図の生成
        generateScatterPlotImage(clusterResults, "クラスタリング：がく片の長さ と がく片の幅", "がく片の長さ(cm)", "がく片の幅(cm)", 0, 1, "sepal_length-sepal_width-scatter.png");
        generateScatterPlotImage(clusterResults, "クラスタリング：がく片の長さ と 花びらの長さ", "がく片の長さ(cm)", "花びらの長さ(cm)", 0, 2, "sepal_length-petal_length-scatter.png");
        generateScatterPlotImage(clusterResults, "クラスタリング：がく片の長さ と 花びらの幅", "がく片の長さ(cm)", "花びらの幅(cm)", 0, 3, "sepal_length-petal_width-scatter.png");
        generateScatterPlotImage(clusterResults, "クラスタリング：がく片の幅 と 花びらの長さ", "がく片の幅(cm)", "花びらの長さ(cm)", 1, 2, "sepal_width-petal_length-scatter.png");
        generateScatterPlotImage(clusterResults, "クラスタリング：がく片の幅 と 花びらの幅", "がく片の幅(cm)", "花びらの幅(cm)", 1, 3, "sepal_width-petal_width-scatter.png");
        generateScatterPlotImage(clusterResults, "クラスタリング：花びらの長さ と 花びらの幅", "花びらの長さ(cm)", "花びらの幅(cm)", 2, 3, "petal_length-petal_width-scatter.png");
    }

    /**
     * データセットの読み込み
     *
     * @return アイリスのレコードリスト
     */
    static List<IrisRecord> getData() {
        String dataPath = "./src/main/resources/data/iris/iris.data";
        List<IrisRecord> target = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(dataPath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] data = line.split(",");
                target.add(
                        new IrisRecord(
                                Double.parseDouble(data[0]),
                                Double.parseDouble(data[1]),
                                Double.parseDouble(data[2]),
                                Double.parseDouble(data[3]),
                                IrisClassification.getByDataValue(data[4])
                        )
                );
            }
        } catch (IOException e) {
            System.out.println("ファイル読み込みに失敗");
        }
        return target;
    }

    /**
     * 実際の散布図（花びらの長さ と 花びらの幅）の画像生成
     *
     * @param records アイリスのデータリスト
     */
    private static void generateActualScatterPlotImagePetal(List<IrisRecord> records) {
        XYSeriesCollection data = new XYSeriesCollection();
        for (IrisClassification irisClassification : IrisClassification.values()) {
            XYSeries series = new XYSeries(irisClassification.dataValue);
            var filteredRecords = records.stream().filter(record -> record.irisClassification == irisClassification).toList();
            for (IrisRecord record : filteredRecords) {
                series.add(record.petalLength, record.petalWidth);
            }
            data.addSeries(series);
        }

        JFreeChart chart = ChartFactory.createScatterPlot(
                "実際：花びらの長さ と 花びらの幅",
                "花びらの長さ(cm)",
                "花びらの幅(cm)",
                data,
                PlotOrientation.VERTICAL,
                true,
                false,
                false
        );

        try {
            ChartUtils.saveChartAsPNG(new File("./image/actual-petal_length-petal_width-scatter.png"), chart, 600, 600);
        } catch (IOException e) {
            System.out.println("ファイル書き込みに失敗 : actual-petal_length-petal_width-scatter.png");
        }
    }

    /**
     * 実際の散布図（がく片の長さ と がく片の幅）の画像生成
     *
     * @param records アイリスのデータリスト
     */
    private static void generateActualScatterPlotImageSepal(List<IrisRecord> records) {
        XYSeriesCollection data = new XYSeriesCollection();
        for (IrisClassification irisClassification : IrisClassification.values()) {
            XYSeries series = new XYSeries(irisClassification.dataValue);
            var filteredRecords = records.stream().filter(record -> record.irisClassification == irisClassification).toList();
            for (IrisRecord record : filteredRecords) {
                series.add(record.sepalLength, record.sepalWidth);
            }
            data.addSeries(series);
        }

        JFreeChart chart = ChartFactory.createScatterPlot(
                "実際：がく片の長さ と がく片の幅",
                "がく片の長さ(cm)",
                "がく片の幅(cm)",
                data,
                PlotOrientation.VERTICAL,
                true,
                false,
                false
        );

        try {
            ChartUtils.saveChartAsPNG(new File("./image/actual-sepal_length-sepal_width-scatter.png"), chart, 600, 600);
        } catch (IOException e) {
            System.out.println("ファイル書き込みに失敗 : actual-sepal_length-sepal_width-scatter.png");
        }
    }

    /**
     * クラスタリング結果の散布図画像ファイル作成
     *
     * @param clusterResults クラスタリング結果
     * @param title          散布図のタイトル
     * @param xAxisLabel     x軸のラベル
     * @param yAxisLabel     y軸のラベル
     * @param xIndex         x軸に当てはめるIrisDataPoint.pointsのインデックス
     * @param yIndex         y軸に当てはめるIrisDataPoint.pointsのインデックス
     * @param filename       画像ファイル名
     */
    private static void generateScatterPlotImage(
            List<CentroidCluster<IrisDataPoint>> clusterResults,
            String title,
            String xAxisLabel,
            String yAxisLabel,
            int xIndex,
            int yIndex,
            String filename
    ) {
        XYSeriesCollection data = new XYSeriesCollection();

        for (int i = 0; i < clusterResults.size(); i++) {
            XYSeries series = new XYSeries("クラスター" + i);

            for (IrisDataPoint point : clusterResults.get(i).getPoints()) {
                series.add(point.points[xIndex], point.points[yIndex]);
            }
            data.addSeries(series);
        }

        JFreeChart chart = ChartFactory.createScatterPlot(
                title,
                xAxisLabel,
                yAxisLabel,
                data,
                PlotOrientation.VERTICAL,
                true,
                false,
                false
        );

        try {
            ChartUtils.saveChartAsPNG(new File("./image/" + filename), chart, 600, 600);
        } catch (IOException e) {
            System.out.println("ファイル書き込みに失敗 : " + filename);
        }
    }

    /**
     * アイリスデータセットのレコード
     *
     * @param sepalLength        がく片の長さ(cm)
     * @param sepalWidth         がく片の幅(cm)
     * @param petalLength        花びらの長さ(cm)
     * @param petalWidth         花びらの幅(cm)
     * @param irisClassification 品種
     */
    private record IrisRecord(
            double sepalLength,
            double sepalWidth,
            double petalLength,
            double petalWidth,
            IrisClassification irisClassification
    ) {

    }

    /**
     * 品種
     */
    private enum IrisClassification {
        IrisSetosa(0, "Iris-setosa"),
        IrisVersicolour(1, "Iris-versicolor"),
        IrisVirginica(2, "Iris-virginica");

        private final int id;
        private final String dataValue;

        IrisClassification(int id, String dataValue) {
            this.id = id;
            this.dataValue = dataValue;
        }

        static IrisClassification getByDataValue(String dataValue) {
            return Arrays.stream(IrisClassification.values())
                    .filter(ic -> ic.dataValue.equals(dataValue))
                    .findFirst()
                    .orElse(null);
        }
    }

    /**
     * クラスタリングするためのアイリスデータのポイント
     */
    public static class IrisDataPoint implements Clusterable {
        private final double[] points;
        private final IrisClassification irisClassification;

        public IrisDataPoint(IrisRecord record) {
            this.points = new double[]{
                    record.sepalLength,
                    record.sepalWidth,
                    record.petalLength,
                    record.petalWidth
            };
            this.irisClassification = record.irisClassification;
        }

        public double[] getPoint() {
            return points;
        }

        public String getPointInfo() {
            return Arrays.toString(this.getPoint()) + ":" + this.irisClassification.dataValue;
        }
    }

}