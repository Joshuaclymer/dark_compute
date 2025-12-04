// JavaScript for Dark Compute Model - Detection Likelihood Section

function plotDarkComputeDetectionSection(data) {
    // Plot individual LR components for Speed of Detection section
    if (data.black_project_model) {
        const years = data.black_project_model.years;

        // Helper function to create LR plot
        function plotLRComponent(elementId, lrData, title) {
            if (!lrData) return;

            const traces = [
                {
                    x: years,
                    y: lrData.p75,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: 'transparent' },
                    showlegend: false,
                    hoverinfo: 'skip'
                },
                {
                    x: years,
                    y: lrData.p25,
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tonexty',
                    fillcolor: 'rgba(91, 141, 190, 0.2)',
                    line: { color: 'transparent' },
                    showlegend: false,
                    hoverinfo: 'skip'
                },
                {
                    x: years,
                    y: lrData.median,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#5B8DBE', width: 2 },
                    name: title,
                    showlegend: false
                }
            ];

            const layout = {
                xaxis: {
                    title: 'Year',
                    titlefont: { size: 10 },
                    tickfont: { size: 9 },
                    range: [years[0], years[years.length - 1]]
                },
                yaxis: {
                    title: 'Likelihood Ratio',
                    titlefont: { size: 10 },
                    tickfont: { size: 9 },
                    type: 'log'
                },
                margin: { l: 50, r: 20, t: 10, b: 40 },
                height: 240,
                hovermode: 'x unified',
            };

            Plotly.newPlot(elementId, traces, layout, {responsive: true, displayModeBar: false});
        }

        // Plot individual LR components for first subsection (initial discrepancies) as distributions
        // These are constant over time, so we extract just the first value from each simulation
        if (data.black_project_model.lr_prc_accounting && data.black_project_model.lr_prc_accounting.individual) {
            const lrPrcSamples = data.black_project_model.lr_prc_accounting.individual.map(sim => sim[0]);
            plotPDF('lrPrcAccountingPlot', lrPrcSamples, '#5B9DB5', 'Likelihood Ratio', 12, true, 1/3, 5);
        }
        if (data.black_project_model.lr_sme_inventory && data.black_project_model.lr_sme_inventory.individual) {
            const lrSmeInventorySamples = data.black_project_model.lr_sme_inventory.individual.map(sim => sim[0]);
            plotPDF('lrSmeInventoryPlot', lrSmeInventorySamples, '#5B9DB5', 'Likelihood Ratio', 12, true, 1/3, 5);
        }

        // Plot combined evidence from reported assets as distribution (also constant over time)
        if (data.black_project_model.lr_initial_stock && data.black_project_model.lr_initial_stock.individual &&
            data.black_project_model.lr_diverted_sme && data.black_project_model.lr_diverted_sme.individual) {
            const lrReportedAssetsSamples = data.black_project_model.lr_initial_stock.individual.map((sim, i) =>
                sim[0] * data.black_project_model.lr_diverted_sme.individual[i][0]
            );
            // Plot in both locations
            plotPDF('lrReportedAssetsInitialPlot', lrReportedAssetsSamples, '#5B9DB5', 'Likelihood Ratio', 12, true, 1/3, 5);
            plotPDF('lrReportedAssetsPlot', lrReportedAssetsSamples, '#5B9DB5', 'Likelihood Ratio', 12, true, 1/3, 5);
        }

        // Plot ongoing intelligence
        plotLRComponent('lrOtherIntelPlot', data.black_project_model.lr_other_intel, 'Other Intel');

        // Plot posterior probability
        if (data.black_project_model.posterior_prob_project) {
            const posteriorData = data.black_project_model.posterior_prob_project;

            // Get prior probability from parameters
            const priorProbInput = document.getElementById('black_project_parameters.p_project_exists');
            const priorProb = priorProbInput ? parseFloat(priorProbInput.value) : 0.1;
            const priorOdds = priorProb / (1 - priorProb);

            // Define thresholds from global config
            const thresholds = DETECTION_CONFIG.getThresholds();

            const traces = [
                {
                    x: years,
                    y: posteriorData.p75,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: 'transparent' },
                    showlegend: false,
                    hoverinfo: 'skip'
                },
                {
                    x: years,
                    y: posteriorData.p25,
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tonexty',
                    fillcolor: 'rgba(91, 141, 190, 0.2)',
                    line: { color: 'transparent' },
                    showlegend: false,
                    hoverinfo: 'skip'
                },
                {
                    x: years,
                    y: posteriorData.median,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#5B8DBE', width: 2 },
                    name: 'Posterior Probability',
                    showlegend: false
                }
            ];

            // Add threshold lines
            for (const threshold of thresholds) {
                const thresholdOdds = priorOdds * threshold.multiplier;
                const thresholdProb = thresholdOdds / (1 + thresholdOdds);

                traces.push({
                    x: [years[0], years[years.length - 1]],
                    y: [thresholdProb, thresholdProb],
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: threshold.color, width: 2, dash: 'dash' },
                    name: `Detection (${threshold.multiplier}x)`,
                    showlegend: true
                });
            }

            const layout = {
                xaxis: {
                    title: 'Year',
                    titlefont: { size: 10 },
                    tickfont: { size: 9 },
                    range: [years[0], years[years.length - 1]]
                },
                yaxis: {
                    title: 'Probability',
                    titlefont: { size: 10 },
                    tickfont: { size: 9 },
                    tickformat: '.0%',
                    range: [0, 1]
                },
                margin: { l: 50, r: 20, t: 15, b: 50 },
                height: 240,
                hovermode: 'x unified',
                showlegend: true,
                legend: {
                    x: 0.02,
                    y: 0.98,
                    xanchor: 'left',
                    yanchor: 'top',
                    bgcolor: 'rgba(255, 255, 255, 0.8)',
                    bordercolor: '#ccc',
                    borderwidth: 1,
                    font: { size: 9 }
                }
            };

            Plotly.newPlot('posteriorProbProjectPlot', traces, layout, {responsive: true, displayModeBar: false});
            setTimeout(() => Plotly.Plots.resize('posteriorProbProjectPlot'), 50);
        }
    }

    // Create the historical plots
    createDetectionLatencyPlot();
    createIntelligenceAccuracyPlot();

    // Update parameter displays to set up click handlers
    if (typeof updateParameterDisplays === 'function') {
        updateParameterDisplays();
    }
}

function createDetectionLatencyPlot(elementId = 'detectionLatencyPlot') {
    // Embedded data from Monte Carlo simulations and nuclear case studies
    const data = {"prediction": {"x": [32.00000000000001, 32.937217912591755, 33.901885119423575, 34.894805557065816, 35.91680670783107, 36.96874028938339, 38.05148296454472, 39.16593707189006, 40.313031377740565, 41.49372185018093, 42.70899245574622, 43.95985597944218, 45.24735486878239, 46.57256210254547, 47.93658208497661, 49.34055156617864, 50.78564058945925, 52.27305346642447, 53.8040297806307, 55.37984542063155, 57.00181364328083, 58.671286168177694, 60.389654304165596, 62.15835010882454, 63.978847581922665, 65.85266389382127, 67.7813606498579, 69.76654519176066, 71.80987193717816, 73.91304375844197, 76.0778134017104, 78.3059849476759, 80.59941531505403, 82.96001580810686, 85.38975370948975, 87.89065391975008, 90.46480064484358, 93.1143391330745, 95.84147746290758, 98.64848838314181, 101.53771120697868, 104.51155376156451, 107.57249439463126, 110.72308403990696, 113.96594834301948, 117.30378984966268, 120.7393902578503, 124.27561273613347, 127.9154043097154, 131.66179831644914, 135.51791693476838, 139.48697378565538, 143.57227661081583, 147.77723002929181, 152.10533837481202, 156.56020861624017, 161.14555336356014, 165.86519396189874, 170.72306367616747, 175.72321096897508, 180.86980287454588, 186.1671284714509, 191.6196024570512, 197.23176882662796, 203.00830466026775, 208.95402402065787, 215.07388196504195, 221.37297867467564, 227.85656370522855, 234.53004036167042, 241.39897020129004, 248.46907766859778, 255.74625486597793, 263.2365664640606, 270.94625475591255, 278.88174485925265, 287.04965007103203, 295.456777378837, 304.11013313371467, 313.01692888913885, 322.18458741099363, 331.6207488635737, 341.333277176762, 351.3302665996905, 361.6200484463427, 372.21119803872404, 383.1125418533841, 394.33316487724477, 405.882418178868, 417.76992670147104, 430.00559728418256, 442.59962691822767, 455.56251124492087, 468.90505330254626, 482.63837252941903, 496.7739140306287, 511.3234581161858, 526.2991301185245, 541.7134104975411, 557.5791452415865, 573.9095565730865, 590.7182539677083, 608.0192454962543, 625.8269494987418, 644.1562066003937, 663.0222920795509, 682.4409285978192, 702.4282993030578, 723.0010613161264, 744.176359612636, 765.9718413112695, 788.4056703805755, 811.4965427764996, 835.263702023263, 859.7269552505728, 884.906689700535, 910.8238897180223, 937.5001542386539, 964.9577147889657, 993.2194540137732, 1022.3089247461583, 1052.2503696359863, 1083.0687413533024, 1114.789723383443, 1147.4397514311977, 1181.0460354518589, 1215.6365823275144, 1251.2402192074876, 1287.8866175323772, 1325.606317761706, 1364.430754825801, 1404.3922843231087, 1445.524209484774, 1487.8608089289658, 1531.4373652280738, 1576.2901943125748, 1622.456675736089, 1669.9752838268375, 1718.8856197514635, 1769.2284445179453, 1821.0457129451024, 1874.3806086269942, 1929.2775799213634, 1985.7823769921133, 2043.942089936678, 2103.805188030079, 2165.421560118367, 2228.8425561950944, 2294.121030195494, 2361.3113840440155, 2430.4696129919193, 2501.6533522827294, 2574.9219251844333, 2650.336392428435, 2727.9596030964876, 2807.856246998011, 2890.0929085814246, 2974.7381224244514, 3061.8624303496294, 3151.5384402126083, 3243.8408864122543, 3338.8466921729737, 3436.635033651173, 3537.2874059192413, 3640.887690882137, 3747.522227183059, 3857.279882156547, 3970.2521258889487, 4086.5331074479736, 4206.21973334483, 4329.411748294443, 4456.2118183409, 4586.725616417528, 4721.061910412836, 4859.332653815766, 5001.653079015706, 5148.141793335179, 5298.920877875049, 5454.115989254762, 5613.856464332342, 5778.2754279914525, 5947.509904085273, 6121.700929629818, 6300.993672341648, 6485.537551618078, 6675.486363060666, 6870.998406645752, 7072.236618648803, 7279.368707432705, 7492.567293212882, 7712.010051915947, 7937.87986325167, 8170.364963121709, 8409.659100492006, 8655.961698859857, 8909.478022449839, 9170.419347277413, 9439.003137222606, 9715.453225260582, 10000.0], "y_median": [11.432874580749703, 11.26799826173969, 11.106826880510456, 10.949247842958057, 10.795152854069794, 10.644437720811597, 10.49700216554045, 10.352749649301732, 10.211587204414812, 10.073425275790791, 9.938177570463408, 9.80576091484879, 9.676095119281658, 9.549102849405378, 9.424709504020605, 9.302843099022928, 9.183434157083601, 9.066415602749414, 8.951722662658318, 8.839292770586423, 8.72906547705968, 8.62098236328009, 8.514986959131695, 8.41102466504582, 8.309042677518482, 8.208989918085349, 8.110816965571185, 8.014475991441646, 7.919920698095468, 7.827106259944446, 7.735989267137639, 7.646527671794422, 7.558680736618818, 7.4724089857748135, 7.38767415790918, 7.304439161214709, 7.22266803043276, 7.142325885699651, 7.063378893146662, 6.985794227168416, 6.909540034279039, 6.834585398479834, 6.760900308066365, 6.688455623806694, 6.617223048426093, 6.547175097337074, 6.478285070556695, 6.41052702575622, 6.343875752390967, 6.278306746861001, 6.213796188655717, 6.150320917437885, 6.087858411024894, 6.026386764227123, 5.965884668505303, 5.906331392410758, 5.847706762774063, 5.789991146609467, 5.7331654337039675, 5.677211019861479, 5.622109790773923, 5.567844106492501, 5.514396786473602, 5.461751095175103, 5.409890728179903, 5.35879979882468, 5.308462825312825, 5.258864718291583, 5.209990768874248, 5.161826637089264, 5.114358340738788, 5.067572244650212, 5.0214550503047315, 4.9759937858279315, 4.931175796327895, 4.8869887345670975, 4.843420551954884, 4.800459489847962, 4.758094071146868, 4.716313092176908, 4.675105614842577, 4.63446095904492, 4.594368695351776, 4.554818637911281, 4.515800837599374, 4.477305575392515, 4.439323355957126, 4.401844901447679, 4.36486114550567, 4.3283632274520265, 4.292342486665847, 4.256790457142621, 4.2216988622253915, 4.187059609502587, 4.1528647858664725, 4.119106652726479, 4.085777641371818, 4.052870348478111, 4.020377531752863, 3.988292105714951, 3.9566071376033327, 3.9253158434105235, 3.8944115840364457, 3.863887861558501, 3.8337383156138323, 3.8039567198899418, 3.7745369787199325, 3.745473123778818, 3.7167593108774803, 3.688389816850951, 3.660359036537872, 3.632661479848063, 3.605291768915265, 3.5782446353322297, 3.5515149174654326, 3.5250975578467916, 3.4989876006398544, 3.47318018917804, 3.4476705635725855, 3.4224540583879315, 3.3975261003823944, 3.3728822063120036, 3.3485179807955, 3.324429114238551, 3.3006113808152917, 3.2770606365053894, 3.2537728171848883, 3.2307439367691395, 3.2079700854061977, 3.185447427719121, 3.163172201095648, 3.141140714023805, 3.119349344472021, 3.097794538312398, 3.076472807785816, 3.0553807300076086, 3.0345149455125786, 3.013872156838172, 2.9934491271446615, 2.9732426788712414, 2.9532496924269545, 2.933467104915429, 2.9138919088924213, 2.894521151155198, 2.8753519315628298, 2.8563814018864853, 2.837606764688862, 2.8190252722319036, 2.8006342254119887, 2.7824309727218, 2.7644129092381142, 2.7465774756347594, 2.72892215722004, 2.711444482997924, 2.694142024752329, 2.6770123961538492, 2.6600532518883035, 2.643262286806485, 2.626637235094528, 2.6101758694643227, 2.5938760003634136, 2.5777354752038586, 2.561752177609515, 2.545924026681261, 2.530248976279648, 2.5147250143245308, 2.4993501621111984, 2.48412247364257, 2.469040034977021, 2.454100963591425, 2.439303407758993, 2.4246455459415373, 2.4101255861957545, 2.3957417655931774, 2.381492349653421, 2.367375631790387, 2.353389932771069, 2.3395336001866522, 2.3258050079355788, 2.312202555718258, 2.2987246685431404, 2.2853697962438546, 2.2721364130071136, 2.259023016911134, 2.246028129474291, 2.2331502952137443, 2.22038808121379, 2.2077400767036996, 2.1952048926447834, 2.1827811613264783, 2.170467535971213, 2.1582626903478404, 2.146165318393419, 2.1341741338431524, 2.1222878698682575, 2.1105052787215977, 2.0988251313908717, 2.087246217259176, 2.075767343772765, 2.064387336115839], "y_low": [1.8711162749107095, 1.7763170635944134, 1.579035124468119, 1.6379061985015222, 1.7369308163084287, 1.683712802416434, 1.528194542559867, 1.4963666215760627, 1.5098915989394022, 1.5621197482380238, 1.4872095524979143, 1.433148844770361, 1.4205920156356953, 1.3163044525139738, 1.2678232962205047, 1.2691846338233772, 1.1919298720628806, 1.163702691284206, 1.1978239223349703, 1.0828426721294178, 1.1154005349483607, 1.2088527714388424, 1.071405598598262, 1.0140497678367477, 0.9939839967676152, 1.0669276942083672, 0.9510749792493828, 0.9885909039481887, 1.0007354958134493, 0.9868692751518544, 0.9364279642368438, 0.8743484413791004, 0.8304084946544052, 0.7889029911962081, 0.8639892803756956, 0.7889353306557066, 0.8226275590721269, 0.785826983828472, 0.7262059719609129, 0.754933487487711, 0.6897999972985237, 0.6401992334710239, 0.6470447333164627, 0.6136502576029512, 0.5483096297560762, 0.5820877956557283, 0.5886312299821125, 0.5244425539555794, 0.5609008988258053, 0.5101388326112021, 0.51260356623569, 0.49912158756898034, 0.47324512396483776, 0.4640469757573988, 0.5211748906486348, 0.4628997665130087, 0.42805769231589874, 0.43284796514501095, 0.3921675219322203, 0.4382220573855957, 0.3657463485060067, 0.3638576212527603, 0.3465965473573856, 0.3153650368247427, 0.32199917202866246, 0.34027297583624955, 0.3211917877062482, 0.3139915408945783, 0.3064628583784539, 0.3000763221974994, 0.26609831170141235, 0.25280243950350206, 0.2610946391114245, 0.2339319568063956, 0.24228974213257487, 0.26016323456283336, 0.2640767528295875, 0.23099864769705505, 0.21207478244028785, 0.22159398797659305, 0.23459127405975475, 0.16736378412325226, 0.17168171787736872, 0.19546575226805452, 0.16779140417058339, 0.15199981569173662, 0.1628712301179875, 0.15451575287609173, 0.16620340268273967, 0.14003097519319774, 0.12300344313933695, 0.14972925396875064, 0.12772109144592786, 0.13937948647610918, 0.12702557217792945, 0.10645874057865745, 0.13118362060378963, 0.10858141162187371, 0.11113264476628995, 0.09751163880381257, 0.0903912719113375, 0.08614734636896562, 0.07727402497077007, 0.08630611308342308, 0.0758292263457196, 0.10514637915020358, 0.0835910962195173, 0.08066770057610892, 0.07655663694826238, 0.07209088431142975, 0.06278221338715553, 0.0756621029533591, 0.057723633654191225, 0.060820649040933934, 0.06409597341319251, 0.058713423843639895, 0.04442066230916305, 0.053980125098437595, 0.04674835772095765, 0.04466715186057374, 0.056704018822014424, 0.04763105396611653, 0.035099904682449244, 0.04101541250821394, 0.03791383565359552, 0.030072789876414947, 0.03562492313603999, 0.03272579842627342, 0.03440792788941216, 0.023646517685686784, 0.028498349305531516, 0.026369084743368305, 0.022507821453364683, 0.019250823844837592, 0.028144225772110048, 0.02004933806375738, 0.023340731798036047, 0.02493182849911202, 0.020309395658979493, 0.020746783101733408, 0.012722607702591822, 0.019724552588411524, 0.014807692477716699, 0.01764455126752592, 0.014227752213058576, 0.012932154711290798, 0.013857007387081094, 0.011692875864218586, 0.011780314333979284, 0.012791358492278962, 0.010082248856083791, 0.012421523564110677, 0.011642705258522882, 0.011976859121951126, 0.011014807726633349, 0.007084390356277525, 0.00991954080821738, 0.008761173903286646, 0.006387978755092498, 0.011144098843200568, 0.006335953650815195, 0.006569793850251726, 0.007967790452795501, 0.008473536485575852, 0.007318594328926014, 0.005761737521359819, 0.003913781713821864, 0.005824354450933688, 0.003101995141905941, 0.004992315786390539, 0.003949095363400306, 0.0032930718635554466, 0.003813472450560078, 0.0030566952909959876, 0.004030136827676397, 0.002686576778623871, 0.004111135876967236, 0.0031486166062476397, 0.002878136918974479, 0.0030998292478788617, 0.002069151412849243, 0.0020817334213851066, 0.002825747171304819, 0.0029325973681639528, 0.0014044172670494483, 0.0012795828429557502, 0.0021239005821701283, 0.002391456242634942, 0.0014490725389072153, 0.0013875850559436358, 0.0014491624617121882, 0.0015818454079577382, 0.0009265533589715661, 0.0007954864413367732, 0.0007906080802144394, 0.0008776885738787605, 0.0009148435339278023, 0.0009697321440422223, 0.0012395730242059436, 0.0006291406751847566], "y_high": [23.822067106013552, 22.777222700316404, 23.565159215781115, 22.956361984691764, 22.192754426670103, 21.80427678000157, 21.802052574658244, 21.15848459899577, 21.689298599668653, 21.780888518408293, 20.575395971581543, 20.34390570286168, 20.186592113321446, 19.97557853499849, 19.53785373531523, 20.062508509851064, 19.069782439702035, 18.68186487458206, 19.148959359445662, 18.5677264078035, 18.371213843816324, 18.086322210692583, 18.13469801863946, 18.405396294767783, 17.81000028271677, 17.07057011828217, 17.4188294388748, 16.979555884531173, 17.41809808402538, 17.568988719405972, 17.30851557397147, 16.571844479839424, 16.624775820955595, 17.095617022889453, 16.65100344998652, 15.971405702751685, 16.253122412917484, 15.919530641411562, 15.873703287997165, 15.976348428036445, 15.741958363419418, 15.218724228335834, 15.593709552860288, 15.575252085060233, 15.076582018837597, 14.86422429565973, 14.992388630524358, 14.568738585716966, 13.953789958715113, 14.654963825528782, 14.619899646788244, 14.114704225763608, 13.950245884449773, 13.532574385875794, 14.569143934041332, 13.933417944396247, 14.230361190093818, 13.753125389162816, 13.680417390574194, 13.619445890176655, 13.963237761903484, 13.430365076436185, 13.502626053726868, 13.078683193785837, 13.313426598192798, 12.957029031337225, 13.12536432797929, 12.74889719513023, 12.570314566462638, 13.125329532593025, 12.370308187154446, 12.845191193414696, 12.830461291616226, 12.675657357674167, 12.777155533041718, 12.495667668296889, 12.488374912829267, 12.344294091406951, 12.137285134006241, 12.628610064579412, 11.562093592259735, 12.10895963038183, 11.786777613130752, 11.99348294657225, 11.960815002847315, 11.925946683596067, 11.301792395640485, 11.67186424137962, 11.451201160598098, 11.95467725264716, 11.529957481186283, 11.836770533689364, 10.871339157027498, 11.70501930715672, 11.13441238001119, 11.685365948059122, 11.229975912074515, 10.746708130073396, 11.012562416280419, 11.566121754718424, 10.498540378695699, 11.698446691996716, 10.784632837918783, 10.938561741401475, 10.406605277815263, 10.947429915224811, 10.864519173756518, 10.480287741135173, 10.805635876164517, 10.662593010655346, 10.310407888476606, 10.622348296648124, 10.447676003908505, 10.685959348119924, 10.480796900885982, 10.285343764168875, 10.516304260839117, 10.602059387344386, 9.734615570376649, 9.900970075237627, 10.256484695558846, 10.597687834159336, 9.593646631417629, 10.233057112225021, 10.401392927955952, 9.903854864849377, 10.220805613740838, 10.371568944234618, 9.546092940484112, 9.928531652912968, 10.09806292086941, 9.466209642563541, 10.023230400522426, 9.965531868707023, 9.626504233890985, 9.977410376593996, 9.64644967629683, 10.025151787824667, 9.51120649848309, 9.795881481035162, 9.117205839378197, 9.349966115270119, 9.487533241738037, 9.105377445300785, 9.975744537322296, 8.983512192808366, 9.854172158004499, 9.002686372520248, 9.089971182305742, 9.085512498334829, 9.29395145819098, 8.803868406410228, 9.07097933364207, 9.180136516315217, 8.733059798470352, 8.781507200236844, 8.92635496811087, 8.990550751949659, 9.138872308108855, 9.080203631485523, 8.736706207307476, 8.408832123069727, 8.733489882589089, 8.729963530670307, 8.123646217557553, 9.610094575366952, 8.964229026490418, 8.711994232571435, 8.708188367746908, 9.045878128253676, 8.39362181309137, 8.390195460369563, 8.754149034820173, 8.48954771694612, 8.392252975502785, 8.942846381390975, 8.53523074819038, 8.280270989695794, 8.340721868647874, 8.914970222387486, 8.375979713727702, 8.493936685258102, 8.456674545740137, 8.787607091137788, 7.95291493970346, 7.524405664420919, 7.910862982442944, 7.976917870935373, 8.388311815232559, 8.471020163743582, 8.16837739472911, 8.592663647523233, 7.848918400894465, 8.160257789355072, 8.288316444281088, 8.394200748750208, 8.62862711472416, 7.656386443444921, 8.054720754450635, 7.850512932849779]}, "actual_data": {"x": [150.0, 115.0, 70.0, 60.0, 350.0, 40.0, 2000.0, 500.0, 3500.0, 200.0, 65.0, 1000.0], "y": [1.0, 6.0, 7.0, 1.5, 1.0, 6.0, 3.0, 5.0, 4.0, 6.0, 9.0, 1.0], "sites": ["Iranian Fordow nuclear enrichment", "Syrian Al-Kibar nuclear reactor", "North Korean 2010 Yongbyon enrichment", "Saudi enrichment facility (unconfirmed)", "Iranian Natanz nuclear enrichment plant", "Libya centrifuge equipment", "Pakistan Kahuta (KRL)", "North Korea 1980s Yongbyon plutonium", "Iraq PC-3 clandestine program", "South Africa covert weapons program", "Iran – Lavizan-Shian site", "Israel – Dimona (Negev NRC)"]}};

    const colors = {
        line: '#5B8DBE',
        fill: 'rgba(91, 141, 190, 0.2)',
        points: '#5B8DBE'
    };

    // Shorten site names to minimum identifiable info
    const shortenSiteName = (name) => {
        const shortNames = {
            'Iranian Fordow nuclear enrichment': 'Iran Fordow',
            'Syrian Al-Kibar nuclear reactor': 'Syria Al-Kibar',
            'North Korean 2010 Yongbyon enrichment': 'North Korea 2010',
            'Saudi enrichment facility (unconfirmed)': 'Saudi facility (unconfirmed)',
            'Iranian Natanz nuclear enrichment plant': 'Iran Natanz',
            'Libya centrifuge equipment': 'Libya',
            'Pakistan Kahuta (KRL)': 'Pakistan Kahuta',
            'North Korea 1980s Yongbyon plutonium': 'North Korea 1980s',
            'Iraq PC-3 clandestine program': 'Iraq PC-3',
            'South Africa covert weapons program': 'South Africa',
            'Iran – Lavizan-Shian site': 'Iran Lavizan-Shian',
            'Israel – Dimona (Negev NRC)': 'Israel Dimona'
        };
        return shortNames[name] || name;
    };

    // Sort actual data points for label positioning
    const pointsData = data.actual_data.x.map((x, i) => ({
        x: x,
        y: data.actual_data.y[i],
        site: shortenSiteName(data.actual_data.sites[i]),
        fullSite: data.actual_data.sites[i]
    })).sort((a, b) => {
        const logDiff = Math.log10(a.x) - Math.log10(b.x);
        if (Math.abs(logDiff) < 0.01) return a.y - b.y;
        return logDiff;
    });

    // Smart label positioning to avoid overlaps
    const labelPositions = [];
    const labelTraces = [];
    const connectorLines = [];

    pointsData.forEach(point => {
        let offsetY = point.y;

        // Try to find non-overlapping position
        for (let iteration = 0; iteration < 100; iteration++) {
            let conflict = false;
            for (const prev of labelPositions) {
                const xLogDist = Math.abs(Math.log10(point.x) - Math.log10(prev.x));
                const yDist = Math.abs(offsetY - prev.y);

                if (xLogDist < 0.6 && yDist < 1.8) {
                    conflict = true;
                    offsetY += 0.9;
                    break;
                }
            }
            if (!conflict) break;
        }

        labelPositions.push({ x: point.x, y: offsetY });
        console.log(`Label for ${point.site}: x=${point.x}, y=${point.y}, offsetY=${offsetY}`);

        // Add connector line if label was moved or if label position is different
        // Always show connector line to make it clear which point the label refers to
        if (Math.abs(offsetY - point.y) > 0.2 || true) {
            connectorLines.push({
                x: [point.x, point.x * 1.15],
                y: [point.y, offsetY],
                mode: 'lines',
                line: { color: 'rgba(0,0,0,0.4)', width: 0.5 },
                showlegend: false,
                hoverinfo: 'skip'
            });
        }

        // Add text label as a scatter trace (not annotation) for better rendering
        const xOffset = point.x * 1.18;
        labelTraces.push({
            x: [xOffset],
            y: [offsetY],
            mode: 'text',
            text: [point.site],
            textposition: 'middle right',
            textfont: { size: 9, color: '#333' },
            showlegend: false,
            hoverinfo: 'skip',
            cliponaxis: false
        });
    });

    // Create traces - use actual values, not log10, since Plotly handles log scale
    const traces = [
        // Confidence interval upper bound
        {
            x: data.prediction.x,
            y: data.prediction.y_high,
            mode: 'lines',
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip',
            name: ''
        },
        // Confidence interval lower bound with fill
        {
            x: data.prediction.x,
            y: data.prediction.y_low,
            mode: 'lines',
            line: { width: 0 },
            fill: 'tonexty',
            fillcolor: colors.fill,
            name: '90% Confidence Interval',
            hoverinfo: 'skip'
        },
        // Median line
        {
            x: data.prediction.x,
            y: data.prediction.y_median,
            mode: 'lines',
            line: { color: colors.line, width: 2 },
            name: 'Posterior Mean',
            hovertemplate: 'workers: %{x}<br>years: %{y:.1f}<extra></extra>'
        },
        // Actual data points
        {
            x: pointsData.map(p => p.x),
            y: pointsData.map(p => p.y),
            mode: 'markers',
            marker: {
                size: 8,
                color: colors.points,
                line: { color: 'white', width: 1 }
            },
            name: '',
            text: pointsData.map(p => p.fullSite),
            hovertemplate: '%{text}<br>workers: %{x}<br>years: %{y:.1f}<extra></extra>',
            showlegend: false
        },
        // Add connector lines and label traces
        ...connectorLines,
        ...labelTraces
    ];

    const layout = {
        xaxis: {
            title: { text: 'Nuclear-role workers', font: { size: 11 } },
            type: 'log',
            tickfont: { size: 9 },
            tickvals: [100, 1000, 10000],
            ticktext: ['100', '1,000', '10,000'],
            gridcolor: 'rgba(128, 128, 128, 0.2)',
            range: [Math.log10(20), Math.log10(15000)]
        },
        yaxis: {
            title: { text: 'Detection latency (years)', font: { size: 11 } },
            tickfont: { size: 9 },
            rangemode: 'tozero',
            gridcolor: 'rgba(128, 128, 128, 0.2)'
        },
        showlegend: true,
        legend: {
            x: 0.02,
            y: 0.98,
            xanchor: 'left',
            yanchor: 'top',
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            bordercolor: '#ccc',
            borderwidth: 1,
            font: { size: 9 }
        },
        hovermode: 'closest',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        margin: { l: 50, r: 20, t: 10, b: 70 },
        height: 320
    };

    Plotly.newPlot(elementId, traces, layout, { responsive: true, displayModeBar: false });
    setTimeout(() => Plotly.Plots.resize(elementId), 50);
}

function createIntelligenceAccuracyPlot(elementId = 'intelligenceAccuracyPlot') {

  // Website color scheme
  const COLORS = {
    'purple': '#9B72B0',
    'blue': '#5B8DBE',
    'teal': '#5AA89B',
    'dark_teal': '#2D6B61',
    'red': '#E74C3C',
    'purple_alt': '#8E44AD',
    'light_teal': '#74B3A8'
  };

  // Helper functions
  function median(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  function linspace(start, end, num) {
    const arr = [];
    const step = (end - start) / (num - 1);
    for (let i = 0; i < num; i++) {
      arr.push(start + step * i);
    }
    return arr;
  }

  // Data for stated error bars (excluding Russian Federation nuclear warheads with min: 1000, max: 2000)
  const statedErrorBars = [
    {"category": "Nuclear Warheads", "min": 150, "max": 160, "date": "1984"},
    {"category": "Nuclear Warheads", "min": 140, "max": 157, "date": "1999"},
    {"category": "Nuclear Warheads", "min": 225, "max": 300, "date": "1984"},
    {"category": "Nuclear Warheads", "min": 60, "max": 80, "date": "1999"},
    {"category": "Fissile material (kg)", "min": 25, "max": 35, "date": "1994"},
    {"category": "Fissile material (kg)", "min": 30, "max": 50, "date": "2007"},
    {"category": "Fissile material (kg)", "min": 17, "max": 33, "date": "1994"},
    {"category": "Fissile material (kg)", "min": 335, "max": 400, "date": "1998"},
    {"category": "Fissile material (kg)", "min": 330, "max": 580, "date": "1996"},
    {"category": "Fissile material (kg)", "min": 240, "max": 395, "date": "2000"},
    {"category": "ICBM launchers", "min": 10, "max": 25, "date": "1961"},
    {"category": "ICBM launchers", "min": 10, "max": 25, "date": "1961"},
    {"category": "ICBM launchers", "min": 105, "max": 120, "date": "1963"},
    {"category": "ICBM launchers", "min": 200, "max": 240, "date": "1964"},
    {"category": "Intercontinental missiles", "min": 180, "max": 190, "date": "2019"},
    {"category": "Intercontinental missiles", "min": 200, "max": 300, "date": "2025"},
    {"category": "Intercontinental missiles", "min": 192, "max": 192, "date": "2024"}
  ];

  // Calculate central estimates and bounds
  const centralEstimates = [];
  const lowerBounds = [];
  const upperBounds = [];
  const categories = [];
  const dates = [];
  const upperPercentErrors = [];
  const lowerPercentErrors = [];

  statedErrorBars.forEach(entry => {
    const central = (entry.min + entry.max) / 2;
    centralEstimates.push(central);
    lowerBounds.push(entry.min);
    upperBounds.push(entry.max);
    categories.push(entry.category);
    dates.push(entry.date);

    const upperError = ((entry.max - central) / central) * 100;
    const lowerError = ((central - entry.min) / central) * 100;
    upperPercentErrors.push(upperError);
    lowerPercentErrors.push(lowerError);
  });

  // Calculate median percent errors
  const medianUpperError = median(upperPercentErrors);
  const medianLowerError = median(lowerPercentErrors);

  // Calculate slopes
  const upperSlope = 1 + (medianUpperError / 100);
  const lowerSlope = 1 - (medianLowerError / 100);

  // Data for estimate vs reality
  const estimates = [700, 800, 900, 300, 1000, 50, 800, 441, 18, 1000, 600, 428.0, 287.0, 311.0, 208];
  const groundTruths = [610, 280, 847, 0, 1308, 60, 819, 499, 5, 1027.1, 661.2, 347.5, 308.0, 247.5, 287];
  const estimateCategories = [
    "Aircraft", "Aircraft", "Aircraft",
    "Chemical Weapons (metric tons)", "Chemical Weapons (metric tons)",
    "Missiles / Launchers", "Missiles / Launchers", "Missiles / Launchers", "Missiles / Launchers",
    "Nuclear Warheads (/10)", "Nuclear Warheads (/10)",
    "Ground combat systems (/10)", "Ground combat systems (/10)", "Ground combat systems (/10)",
    "Troops (/1000)"
  ];
  const estimateDates = [
    "1991", "1956", "2003",  // Aircraft
    "2003", "2003",  // Chemical Weapons
    "1956", "2003", "2003", "1960",  // Missiles / Launchers
    "1980s", "1980s",  // Nuclear Warheads
    "1991", "2003", "2003",  // Ground combat systems
    "1991"  // Troops
  ];

  // Calculate median estimate error
  const estimatePercentErrors = [];
  for (let i = 0; i < estimates.length; i++) {
    if (groundTruths[i] !== 0) {
      estimatePercentErrors.push(Math.abs((estimates[i] - groundTruths[i]) / groundTruths[i]) * 100);
    }
  }
  const medianEstimateError = median(estimatePercentErrors);

  const estimateUpperSlope = 1 + (medianEstimateError / 100);
  const estimateLowerSlope = 1 - (medianEstimateError / 100);

  // Labels for specific points
  const labels = [
    {"index": 8, "label": "Missile gap"},
    {"index": 1, "label": "Bomber gap"},
    {"index": 3, "label": "Iraq intelligence failure"}
  ];

  // Create traces for left subplot (Stated ranges)
  const leftTraces = [];
  const pointColor = '#5B9DB5'; // Blue-green

  // Add error bar lines
  for (let i = 0; i < centralEstimates.length; i++) {
    leftTraces.push({
      x: [centralEstimates[i], centralEstimates[i]],
      y: [lowerBounds[i], upperBounds[i]],
      mode: 'lines',
      line: { color: pointColor, width: 1 },
      opacity: 0.3,
      showlegend: false,
      hoverinfo: 'skip',
      xaxis: 'x',
      yaxis: 'y'
    });
  }

  // Add upper bound points
  leftTraces.push({
    x: centralEstimates,
    y: upperBounds,
    mode: 'markers',
    marker: { color: pointColor, size: 8 },
    showlegend: false,
    customdata: dates.map((d, i) => [categories[i], d]),
    hovertemplate: '%{customdata[0]}<br>Date: %{customdata[1]}<br>Central: %{x}<br>Upper: %{y}<extra></extra>',
    xaxis: 'x',
    yaxis: 'y'
  });

  // Add lower bound points
  leftTraces.push({
    x: centralEstimates,
    y: lowerBounds,
    mode: 'markers',
    marker: { color: pointColor, size: 8 },
    showlegend: false,
    customdata: dates.map((d, i) => [categories[i], d]),
    hovertemplate: '%{customdata[0]}<br>Date: %{customdata[1]}<br>Central: %{x}<br>Lower: %{y}<extra></extra>',
    xaxis: 'x',
    yaxis: 'y'
  });

  // Add median error region for left subplot
  const maxRange = Math.max(...centralEstimates, ...upperBounds);
  const xLine = linspace(0, maxRange, 100);

  leftTraces.push({
    x: [...xLine, ...[...xLine].reverse()],
    y: [...xLine.map(x => lowerSlope * x), ...[...xLine].reverse().map(x => upperSlope * x)],
    fill: 'toself',
    fillcolor: 'lightgray',
    opacity: 0.3,
    line: { width: 0 },
    name: `Median error margin = ${medianUpperError.toFixed(1)}%`,
    showlegend: true,
    hoverinfo: 'skip',
    xaxis: 'x',
    yaxis: 'y',
    legend: 'legend'
  });

  // Add y=x line for left subplot
  leftTraces.push({
    x: xLine,
    y: xLine,
    mode: 'lines',
    line: { color: 'grey', width: 1 },
    opacity: 0.5,
    showlegend: false,
    hoverinfo: 'skip',
    xaxis: 'x',
    yaxis: 'y'
  });

  // Create traces for right subplot (Estimate vs reality)
  // These will be adjusted for narrow screens later
  const rightTracesTemplate = [];

  // Add all estimate points with same color
  rightTracesTemplate.push({
    x: groundTruths,
    y: estimates,
    mode: 'markers',
    marker: { color: pointColor, size: 8 },
    showlegend: false,
    customdata: estimateDates.map((d, i) => [estimateCategories[i], d]),
    hovertemplate: '%{customdata[0]}<br>Date: %{customdata[1]}<br>Ground truth: %{x}<br>Estimate: %{y}<extra></extra>'
  });

  // Add median error region for right subplot
  const maxRangeEst = Math.max(...estimates, ...groundTruths);
  const xLineEst = linspace(0, maxRangeEst, 100);

  rightTracesTemplate.push({
    x: [...xLineEst, ...[...xLineEst].reverse()],
    y: [...xLineEst.map(x => estimateLowerSlope * x), ...[...xLineEst].reverse().map(x => estimateUpperSlope * x)],
    fill: 'toself',
    fillcolor: 'lightgray',
    opacity: 0.3,
    line: { width: 0 },
    name: `Median error margin = ${medianEstimateError.toFixed(1)}%`,
    showlegend: true,
    hoverinfo: 'skip'
  });

  // Add y=x line for right subplot
  rightTracesTemplate.push({
    x: xLineEst,
    y: xLineEst,
    mode: 'lines',
    line: { color: 'grey', width: 1 },
    opacity: 0.5,
    showlegend: false,
    hoverinfo: 'skip'
  });

  // Check if screen is narrow
  const containerWidth = document.getElementById(elementId)?.offsetWidth || window.innerWidth;
  const isNarrow = containerWidth < 600;

  // Adjust right traces for axis references
  const rightTraces = rightTracesTemplate.map(trace => {
    if (isNarrow) {
      // For narrow screens, use primary axes
      return trace;
    } else {
      // For wide screens, use secondary axes
      return { ...trace, xaxis: 'x2', yaxis: 'y2', legend: 'legend2' };
    }
  });

  // Combine traces based on screen width
  const data = isNarrow ? rightTraces : [...leftTraces, ...rightTraces];

  // Create layout with subplots (or single plot for narrow screens)
  const layout = isNarrow ? {
    // Single plot layout for narrow screens
    height: 320,
    showlegend: true,
    autosize: true,
    xaxis: {
      title: 'Ground Truth',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray'
    },
    yaxis: {
      title: 'Estimate',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray'
    },
    legend: {
      font: { size: 9 },
      orientation: 'v',
      yanchor: 'top',
      y: 0.98,
      xanchor: 'left',
      x: 0.02,
      bgcolor: 'rgba(255, 255, 255, 0.8)',
      bordercolor: 'lightgray',
      borderwidth: 1
    },
    margin: { l: 50, r: 20, t: 40, b: 70 },
    plot_bgcolor: 'white',
    font: { size: 10 },
    annotations: [
      {
        text: 'Estimate vs. ground truth',
        xref: 'paper',
        yref: 'paper',
        x: 0.5,
        y: 1.05,
        xanchor: 'center',
        yanchor: 'bottom',
        showarrow: false,
        font: { size: 12 }
      }
    ]
  } : {
    // Two subplot layout for wider screens
    height: 320,
    showlegend: true,
    autosize: true,
    grid: {
      rows: 1,
      columns: 2,
      pattern: 'independent',
      xgap: 0.12
    },
    xaxis: {
      title: 'Central Estimate (Midpoint)',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray',
      domain: [0, 0.44]
    },
    yaxis: {
      title: 'Stated estimate range',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray'
    },
    xaxis2: {
      title: 'Ground Truth',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray',
      domain: [0.56, 1]
    },
    yaxis2: {
      title: 'Estimate',
      showgrid: true,
      gridwidth: 1,
      gridcolor: 'lightgray',
      anchor: 'x2'
    },
    legend: {
      font: { size: 9 },
      orientation: 'v',
      yanchor: 'top',
      y: 0.98,
      xanchor: 'left',
      x: 0.02,
      bgcolor: 'rgba(255, 255, 255, 0.8)',
      bordercolor: 'lightgray',
      borderwidth: 1
    },
    legend2: {
      font: { size: 9 },
      orientation: 'v',
      yanchor: 'top',
      y: 0.98,
      xanchor: 'left',
      x: 0.58,
      bgcolor: 'rgba(255, 255, 255, 0.8)',
      bordercolor: 'lightgray',
      borderwidth: 1
    },
    margin: { l: 50, r: 20, t: 10, b: 70 },
    plot_bgcolor: 'white',
    font: { size: 10 },
    annotations: [
      {
        text: 'Stated ranges',
        xref: 'paper',
        yref: 'paper',
        x: 0.22,
        y: 1.05,
        xanchor: 'center',
        yanchor: 'bottom',
        showarrow: false,
        font: { size: 12 }
      },
      {
        text: 'Estimate vs. ground truth',
        xref: 'paper',
        yref: 'paper',
        x: 0.78,
        y: 1.05,
        xanchor: 'center',
        yanchor: 'bottom',
        showarrow: false,
        font: { size: 12 }
      }
    ]
  };

  // Add text labels for specific points
  const labelOffsets = {
    8: [10, 20],   // Missile gap
    1: [10, -25],  // Bomber gap
    3: [10, -30]    // Iraq intelligence failure
  };

  labels.forEach(labelInfo => {
    const idx = labelInfo.index;
    const offset = labelOffsets[idx] || [10, 10];

    layout.annotations.push({
      x: groundTruths[idx],
      y: estimates[idx],
      text: labelInfo.label,
      showarrow: true,
      arrowhead: 2,
      arrowsize: 1,
      arrowwidth: 1,
      arrowcolor: 'gray',
      ax: offset[0],
      ay: offset[1],
      font: { size: 9 },
      bgcolor: 'white',
      bordercolor: 'gray',
      borderwidth: 1,
      borderpad: 3,
      opacity: 0.8,
      xref: isNarrow ? 'x' : 'x2',
      yref: isNarrow ? 'y' : 'y2'
    });
  });

  const config = {
    responsive: true,
    displayModeBar: false
  };

  Plotly.newPlot(elementId, data, layout, config);
  setTimeout(() => Plotly.Plots.resize(elementId), 50);
}
