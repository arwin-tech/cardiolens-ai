import { PieChart, Pie, Cell } from 'recharts';

interface Props {
  percentage: number;
  category: string;
}

export default function RiskGauge({ percentage, category }: Props) {
  const data = [
    { name: 'Risk', value: percentage },
    { name: 'Remaining', value: 100 - percentage },
  ];

  const getColor = (cat: string) => {
    switch (cat.toLowerCase()) {
      case 'low': return '#6B9E7E';
      case 'moderate': return '#C89968';
      case 'high':
      case 'elevated': return '#A84040';
      default: return '#8B2D2D';
    }
  };

  const riskColor = getColor(category);

  return (
    <div className="flex flex-col items-center justify-center p-6 bg-[#FFFBF8] border border-[#E8DFD9] rounded-medical shadow-sm">
      <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">Cardiovascular Risk Score</h4>
      <div className="relative w-48 h-48 flex items-center justify-center">
        <PieChart width={192} height={192}>
          <Pie
            data={data}
            cx={91}
            cy={91}
            innerRadius={65}
            outerRadius={85}
            startAngle={90}
            endAngle={-270}
            dataKey="value"
            stroke="none"
          >
            <Cell key="cell-risk" fill={riskColor} />
            <Cell key="cell-bg" fill="#E8DFD9" />
          </Pie>
        </PieChart>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-extrabold text-[#2C2C2C]">{percentage.toFixed(1)}%</span>
          <span className="text-xs font-bold uppercase tracking-wide mt-1" style={{ color: riskColor }}>
            {category} Tier
          </span>
        </div>
      </div>
    </div>
  );
}