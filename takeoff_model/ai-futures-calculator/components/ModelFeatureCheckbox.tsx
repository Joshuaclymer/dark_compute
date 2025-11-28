import React from 'react';

interface ModelFeatureCheckboxProps {
  id: string;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  hoverText: string;
  style?: 'checkbox' | 'switch';
  disabled?: boolean;
}

export const ModelFeatureCheckbox: React.FC<ModelFeatureCheckboxProps> = ({
  id,
  label,
  checked,
  onChange,
  hoverText,
  style = 'checkbox',
  disabled = false,
}) => {
  const isSwitch = style === 'switch';

  const inputElement = (
    <input
      type="checkbox"
      id={id}
      checked={checked}
      onChange={(e) => onChange(e.target.checked)}
      disabled={disabled}
      className={isSwitch ? "sr-only peer" : "h-4 w-4 rounded border-gray-300 text-accent focus:ring-accent accent-accent"}
    />
  );

  const labelElement = (
    <label htmlFor={id} className="text-sm font-medium text-gray-900 cursor-pointer">
      {label}
    </label>
  );

  const hoverTextElement = (
    <div className="pointer-events-none absolute left-1/2 -translate-x-1/2 bottom-full mb-2 hidden w-64 rounded-md bg-gray-900 px-3 py-2 text-xs text-white shadow-lg group-hover:block z-50">
      <div className="relative">
        {hoverText}
        <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-x-4 border-x-transparent border-t-4 border-t-gray-900"></div>
      </div>
    </div>
  );

  if (isSwitch) {
    return (
      <div className="group relative flex items-center space-x-2">
        <label className="relative inline-flex items-center cursor-pointer">
          {inputElement}
          <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-[#2A623D] rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-[#2A623D]"></div>
        </label>
        {labelElement}
        {hoverTextElement}
      </div>
    );
  }

  return (
    <div className="group relative flex items-center space-x-2">
      {inputElement}
      {labelElement}
      {hoverTextElement}
    </div>
  );
};

