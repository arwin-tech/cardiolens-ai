import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { SHAPFeature } from '../types/api';

interface Props {
  features: SHAPFeature[];
}

export default function SHAPChart({ features }: Props) {
  const chartData = [...features]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 8);

  return (
    <div className="p-6 bg-[#FFFBF8] border border-[#E8DFD9] rounded-medical shadow-sm space-y-4">
      <h4 className="text-lg font-bold text-[#8B2D2D]">SHAP Local Feature Impact Ranking</h4>
      <p className="text-xs text-gray-500">Positive values increase predicted risk; negative values lower calculated risk.</p>
      
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ left: 30, right: 30 }}>
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis dataKey="feature" type="category" tick={{ fontSize: 11 }} width={80} />
            <Tooltip
              formatter={(val: any) => Number(val).toFixed(4)}
              contentStyle={{ backgroundColor: '#FFFBF8', borderColor: '#E8DFD9', borderRadius: '8px' }}
            />
            <Bar dataKey="shap_value" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.shap_value > 0 ? '#A84040' : '#6B9E7E'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}